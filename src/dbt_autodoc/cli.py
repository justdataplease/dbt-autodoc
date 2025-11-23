import os
import glob
import time
import getpass
import subprocess
import argparse
import re
import sys
import asyncio
from dotenv import load_dotenv, dotenv_values

# --- 1. ROBUST DEPENDENCY CHECK ---
def check_dependencies():
    missing = []
    try:
        import duckdb
    except ImportError:
        missing.append("duckdb")
    try:
        import google.generativeai as genai
    except ImportError:
        missing.append("google-generativeai")
    try:
        from ruamel.yaml import YAML
        from ruamel.yaml.scalarstring import DoubleQuotedScalarString
    except ImportError:
        missing.append("ruamel.yaml")
    
    # Check for asyncpg if postgres is likely to be used, but we'll soft check later or here?
    # The user specifically asked to make postgres async.
    try:
        import asyncpg
    except ImportError:
        pass # We will check this when connecting if postgres is selected

    if missing:
        print("❌ Missing required libraries. Please run:")
        print(f"   pip install {' '.join(missing)}")
        sys.exit(1)


check_dependencies()

# Now safe to import
import duckdb
import google.generativeai as genai
from ruamel.yaml import YAML, YAMLError
from ruamel.yaml.scalarstring import DoubleQuotedScalarString
try:
    import asyncpg
except ImportError:
    asyncpg = None


# --- 2. SETUP & CONFIG ---

def get_dbt_project_info():
    """Extracts project name and profile from dbt_project.yml."""
    try:
        with open("dbt_project.yml", "r", encoding='utf-8') as f:
            data = yaml.load(f)
            return {
                "name": data.get("name", "unknown_project"),
                "profile": data.get("profile", "unknown_profile")
            }
    except Exception:
        return {"name": "unknown_project", "profile": "unknown_profile"}

def load_config(config_path="dbt-autodoc.yml"):
    if not os.path.exists(config_path):
        print(f"⚠️  Config file '{config_path}' not found.")
        print("⚙️  Generating sample config...")
        
        try:
            sample_path = os.path.join(os.path.dirname(__file__), 'sample_config.yml')
            with open(sample_path, 'r', encoding='utf-8') as f:
                sample_config = f.read()

            with open(config_path, 'w', encoding='utf-8') as f:
                f.write(sample_config)
            print(f"✅ Created '{config_path}'.")
            print("❗ Please configure it and run the script again.")
            sys.exit(0)
        except Exception as e:
            print(f"❌ Failed to create config file: {e}")
            sys.exit(1)

    yaml_loader = YAML(typ='safe')
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml_loader.load(f) or {}
    except Exception as e:
        print(f"⚠️  Warning: Could not load config file: {e}")
        return {}


CFG = load_config()

# Constants
DB_TYPE = CFG.get("db_type", "duckdb").lower()
DUCKDB_PATH = CFG.get("duckdb_path", "docs_backup.duckdb")
DBT_MODELS_DIR = CFG.get("dbt_models_dir", "models")
AI_TAG = CFG.get("ai_tag", "(ai_generated)")
COMPANY_CONTEXT = CFG.get("company_context", "")
GEMINI_MODEL_NAME = CFG.get("gemini_model", "gemini-2.5-flash")

# Special marker for Table descriptions in the DB
TABLE_MARKER = "__TABLE__"

# YAML Handling
yaml = YAML()
yaml.preserve_quotes = True
yaml.indent(mapping=2, sequence=4, offset=2)
yaml.width = 4096

# Global Model Placeholder
model = None

# Semaphore for concurrency control
concurrency_sem = None


# --- 3. DATABASE ADAPTER ---

def get_current_user():
    return os.getenv('DBT_USER') or os.getenv('USER') or os.getenv('USERNAME') or getpass.getuser() or 'unknown'


class DatabaseAdapter:
    def __init__(self, project_info=None):
        self.conn = None # For DuckDB
        self.pool = None # For Postgres
        self.type = DB_TYPE
        self.project_name = project_info.get("name", "unknown") if project_info else "unknown"
        self.profile_name = project_info.get("profile", "unknown") if project_info else "unknown"

    async def connect(self):
        try:
            if self.type == 'postgres':
                if not asyncpg:
                    print("❌ DB Type is 'postgres' but 'asyncpg' is not installed.")
                    print("   pip install asyncpg")
                    sys.exit(1)

                postgres_url = os.getenv('POSTGRES_URL')
                
                # Fallback: Try manual parsing if .env exists and env var is missing
                if not postgres_url and os.path.exists(".env"):
                    try:
                        env_config = dotenv_values(".env")
                        if "POSTGRES_URL" in env_config:
                            postgres_url = env_config["POSTGRES_URL"]
                            print("⚠️  Loaded POSTGRES_URL from .env manually (fallback).")
                    except Exception as e:
                        print(f"⚠️  Manual .env parse failed: {e}")

                if not postgres_url:
                    if os.path.exists(".env"):
                        raise ValueError("POSTGRES_URL not found in environment, but .env file exists. Check variable name and format.")
                    else:
                        raise ValueError("POSTGRES_URL environment variable is missing and no .env file found.")
                
                self.pool = await asyncpg.create_pool(postgres_url)
            else:
                self.conn = duckdb.connect(DUCKDB_PATH)
        except Exception as e:
            print(f"❌ CRITICAL: Could not connect to database ({self.type}).")
            if "IO Error" in str(e) and self.type == 'duckdb':
                print("   (Hint: Is another process/DBeaver holding the .duckdb file open?)")
            print(f"   Error details: {e}")
            sys.exit(1)

    async def init_table(self):
        q_cache_pg = """
            CREATE TABLE IF NOT EXISTS doc_cache
            (
                dbt_project_name VARCHAR,
                dbt_profile_name VARCHAR,
                model_name VARCHAR,
                column_name VARCHAR,
                description VARCHAR,
                user_name VARCHAR,
                is_human BOOLEAN,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (dbt_project_name, dbt_profile_name, model_name, column_name)
            )
            """
        q_log_pg = """
            CREATE TABLE IF NOT EXISTS doc_cache_log
            (
                dbt_project_name VARCHAR,
                dbt_profile_name VARCHAR,
                model_name VARCHAR,
                column_name VARCHAR,
                old_description VARCHAR,
                new_description VARCHAR,
                user_name VARCHAR,
                is_human BOOLEAN,
                changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        
        q_cache_duck = q_cache_pg
        q_log_duck = q_log_pg

        for attempt in range(2):
            try:
                if self.type == 'postgres':
                    async with self.pool.acquire() as conn:
                        await conn.execute(q_cache_pg)
                        await conn.execute(q_log_pg)
                else:
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(None, lambda: self.conn.execute(q_cache_duck))
                    await loop.run_in_executor(None, lambda: self.conn.execute(q_log_duck))
                
                if await self.migrate_schema():
                    print("♻️  Schema mismatch detected & handled. Re-initializing tables...")
                    continue # Retry create after drop
                break # Done if no migration needed or handled
            except Exception as e:
                print(f"❌ Error initializing database table: {e}")
                print("   If you have an old database schema, you might need to run with --cleanup-db.")
                sys.exit(1)

    async def migrate_schema(self):
        # Simplified migration: If critical columns missing, DROP tables to reset.
        try:
            tables = ["doc_cache", "doc_cache_log"]
            required_columns = ["dbt_project_name", "dbt_profile_name", "user_name", "is_human"]
            
            needs_reset = False

            for table in tables:
                existing_cols = []
                if self.type == 'postgres':
                    async with self.pool.acquire() as conn:
                        rows = await conn.fetch(f"SELECT column_name FROM information_schema.columns WHERE table_name = '{table}'")
                        existing_cols = [row['column_name'] for row in rows]
                else:
                    loop = asyncio.get_running_loop()
                    def _check_cols():
                        try:
                            info = self.conn.execute(f"PRAGMA table_info('{table}')").fetchall()
                            return [col[1] for col in info]
                        except:
                            return []
                    existing_cols = await loop.run_in_executor(None, _check_cols)
                
                if not existing_cols: continue # Table doesn't exist yet (handled by init)

                for req in required_columns:
                    if req not in existing_cols:
                        print(f"⚠️  Schema mismatch in {table}: Missing {req}. Resetting table (Dev Mode)...")
                        needs_reset = True
                        break
                if needs_reset: break

            if needs_reset:
                await action_cleanup_db(self)
                return True # Signal to re-init

            return False

        except Exception as e:
            print(f"⚠️  Migration Check Warning: {e}")
            return False

    async def get(self, model, col):
        try:
            if self.type == 'duckdb':
                q = "SELECT description FROM doc_cache WHERE dbt_project_name = ? AND dbt_profile_name = ? AND model_name = ? AND column_name = ?"
                params = (self.project_name, self.profile_name, model, col)
                
                def _fetch():
                    res = self.conn.execute(q, params).fetchone()
                    return res[0] if res else None

                loop = asyncio.get_running_loop()
                return await loop.run_in_executor(None, _fetch)
            else:
                q = "SELECT description FROM doc_cache WHERE dbt_project_name = $1 AND dbt_profile_name = $2 AND model_name = $3 AND column_name = $4"
                async with self.pool.acquire() as conn:
                    val = await conn.fetchval(q, self.project_name, self.profile_name, model, col)
                    return val
        except Exception as e:
            print(f"⚠️  DB Read Error ({model}.{col}): {e}")
            return None

    async def save(self, model, col, description):
        if not description: return
        try:
            clean_desc = str(description).strip('"')
            old_desc = await self.get(model, col)
            user = get_current_user()
            is_human = AI_TAG not in clean_desc
            
            # Only log if there is a change
            if old_desc != clean_desc:
                await self.log_change(model, col, old_desc, clean_desc, user, is_human)

            if self.type == 'postgres':
                q = """
                    INSERT INTO doc_cache (dbt_project_name, dbt_profile_name, model_name, column_name, description, user_name, is_human, updated_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, CURRENT_TIMESTAMP)
                    ON CONFLICT (dbt_project_name, dbt_profile_name, model_name, column_name)
                    DO UPDATE SET description = EXCLUDED.description, user_name = EXCLUDED.user_name, is_human = EXCLUDED.is_human, updated_at = CURRENT_TIMESTAMP
                    """
                async with self.pool.acquire() as conn:
                    await conn.execute(q, self.project_name, self.profile_name, model, col, clean_desc, user, is_human)
            else:
                q = """
                    INSERT OR REPLACE INTO doc_cache (dbt_project_name, dbt_profile_name, model_name, column_name, description, user_name, is_human, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, lambda: self.conn.execute(q, (self.project_name, self.profile_name, model, col, clean_desc, user, is_human)))
        except Exception as e:
            print(f"⚠️  DB Save Error ({model}.{col}): {e}")

    async def log_change(self, model, col, old_desc, new_desc, user, is_human):
        try:
            if self.type == 'postgres':
                q = """
                    INSERT INTO doc_cache_log (dbt_project_name, dbt_profile_name, model_name, column_name, old_description, new_description, user_name, is_human, changed_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, CURRENT_TIMESTAMP)
                    """
                async with self.pool.acquire() as conn:
                    await conn.execute(q, self.project_name, self.profile_name, model, col, old_desc, new_desc, user, is_human)
            else:
                q = """
                    INSERT INTO doc_cache_log (dbt_project_name, dbt_profile_name, model_name, column_name, old_description, new_description, user_name, is_human, changed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, lambda: self.conn.execute(q, (self.project_name, self.profile_name, model, col, old_desc, new_desc, user, is_human)))
        except Exception as e:
            print(f"⚠️  DB Log Error ({model}.{col}): {e}")

    async def close(self):
        if self.type == 'postgres' and self.pool:
            await self.pool.close()
        elif self.type == 'duckdb' and self.conn:
            try:
                self.conn.close()
            except:
                pass


class DbtConfigManipulator:
    @staticmethod
    def extract_description(sql_content):
        try:
            start_marker = "{{ config("
            start_idx = sql_content.find(start_marker)
            if start_idx == -1:
                start_marker = "{{config("
                start_idx = sql_content.replace(" ", "").find(start_marker)
                if start_idx == -1:
                    return None

            open_paren_idx = sql_content.find("(", start_idx)
            balance = 0
            insertion_point = -1

            for i in range(open_paren_idx, len(sql_content)):
                char = sql_content[i]
                if char == '(':
                    balance += 1
                elif char == ')':
                    balance -= 1
                if balance == 0:
                    insertion_point = i
                    break

            if insertion_point == -1: return None

            config_body = sql_content[open_paren_idx + 1: insertion_point]
            desc_pattern = re.compile(r"description\s*=\s*(['\"])([\s\S]*?)\1")
            match = desc_pattern.search(config_body)

            if match:
                return match.group(2)
            return None
        except Exception:
            return None

    @staticmethod
    def update_or_create(sql_content, description):
        try:
            clean_desc = str(description).replace('"', "'")
            start_marker = "{{ config("
            start_idx = sql_content.find(start_marker)

            if start_idx == -1:
                start_marker = "{{config("
                start_idx = sql_content.replace(" ", "").find(start_marker)
                if start_idx == -1:
                    new_config = f'{{{{ config(\n    description = "{clean_desc}"\n) }}}}\n\n'
                    return new_config + sql_content

            balance = 0
            insertion_point = -1
            open_paren_idx = sql_content.find("(", start_idx)

            for i in range(open_paren_idx, len(sql_content)):
                char = sql_content[i]
                if char == '(':
                    balance += 1
                elif char == ')':
                    balance -= 1
                if balance == 0:
                    insertion_point = i
                    break

            if insertion_point == -1: return sql_content

            config_body = sql_content[open_paren_idx + 1: insertion_point]
            desc_pattern = re.compile(r"(description\s*=\s*)(['\"])([\s\S]*?)(['\"])")
            match = desc_pattern.search(config_body)

            if match:
                new_body = config_body[:match.start(3)] + description + config_body[match.end(3):]
                return sql_content[:open_paren_idx + 1] + new_body + sql_content[insertion_point:]
            else:
                clean_body = config_body.rstrip()
                needs_comma = True if clean_body and not clean_body.endswith(",") else False
                comma = "," if needs_comma else ""

                if not clean_body:
                    new_body = f'\n    description = "{description}"\n'
                else:
                    new_body = f'{clean_body}{comma}\n    description = "{description}"\n'

                return sql_content[:open_paren_idx + 1] + new_body + sql_content[insertion_point:]
        except Exception as e:
            print(f"❌ parsing error in SQL update: {e}")
            return sql_content


# --- 4. AI HELPER ---

async def ask_gemini(model_name, target_name, is_table=False, table_context=None, sql_content=None, show_prompt=False):
    if not model:
        return None

    entity_type = "Table" if is_table else "Column"
    context_block = f"\n    Parent Table Context: {table_context}\n" if (table_context and not is_table) else ""

    sql_block = ""
    if is_table and sql_content:
        safe_sql = sql_content[:15000]
        sql_block = f"\n    SQL Source Code:\n    ```sql\n{safe_sql}\n    ```\n"

    prompt = f"""
    You are a Data Dictionary Editor. Your goal is to write technical, dry, and precise definitions.

    INPUT CONTEXT:
    - Business Context: {COMPANY_CONTEXT} (Use this to understand the logic, but DO NOT use the company name in the output).
    - Model Name: {model_name}
    - Type: {entity_type}
    - Object Name: {target_name}
    {context_block}
    {sql_block}

    STRICT WRITING RULES:
    1. START IMMEDIATELY with the definition. Do NOT use phrases like "This column represents...", "This is...", "Contains...", or "A field showing...".
    2. FORBIDDEN: Do not use subjective adjectives (e.g., "valuable", "important", "key", "robust", "comprehensive").
    3. FORBIDDEN: Do not mention the company name "{COMPANY_CONTEXT}" in the output. Keep it generic (e.g., use "the platform" or "users" instead of "Amazon users").
    4. LENGTH: Keep it under 25 words.
    5. SYNTAX: 
       - If boolean: Start with "Flag for...".
       - If timestamp: Start with "Date and time when...".
       - If ID: Start with "Unique identifier for...".

    EXAMPLE OUTPUTS:
    - Good: "Total revenue generated from completed orders including tax."
    - Bad: "Key metric showing the amazing Amazon revenue for orders."

    GENERATE DEFINITION FOR {target_name}:
    """

    if show_prompt:
        print(f"\n--- 📝 PROMPT DEBUG ({model_name}.{target_name}) ---")
        print(prompt)
        print("---------------------------------------------------\n")

    try:
        if concurrency_sem:
            async with concurrency_sem:
                print(f"🤖 Asking AI for {model_name} -> {target_name}...")
                response = await model.generate_content_async(prompt)
        else:
            print(f"🤖 Asking AI for {model_name} -> {target_name}...")
            response = await model.generate_content_async(prompt)
            
        if not response.text:
            print("⚠️  AI returned empty text (possibly safety filtered).")
            return None

        text = response.text.strip().strip('"').strip("'")
        return f"{text} {AI_TAG}"
    except Exception as e:
        print(f"⚠️  AI API Error: {e}")
        return None


# --- 5. CENTRAL LOGIC ---

async def resolve_description(current_desc, model_name, col_name, db, use_ai, is_table=False, table_context=None, sql_content=None, show_prompt=False):
    current_desc_str = str(current_desc) if current_desc else ""

    # 1. Keep Human Written
    if current_desc_str and AI_TAG not in current_desc_str:
        await db.save(model_name, col_name, current_desc_str)
        return current_desc_str

    # 2. Keep Existing AI
    if current_desc_str and AI_TAG in current_desc_str:
        await db.save(model_name, col_name, current_desc_str)
        return current_desc_str

    # 3. Restore from DB
    cached_desc = await db.get(model_name, col_name)
    if cached_desc:
        is_human_cached = AI_TAG not in cached_desc

        if is_human_cached:
            print(f"💾 Restored Human Description from DB: {model_name}.{col_name}")
            return cached_desc
        
        if not use_ai:
            print(f"💾 Restored AI Description from DB: {model_name}.{col_name}")
            return cached_desc

    # 4. Ask AI
    if use_ai:
        ai_text = await ask_gemini(model_name, col_name, is_table, table_context, sql_content, show_prompt)
        if ai_text:
            await db.save(model_name, col_name, ai_text)
            # time.sleep(1) # Removed in favor of concurrency limits if needed, or let semaphore handle it
            return ai_text

    return current_desc


# --- 6. ACTIONS ---

def action_cleanup():
    pattern = "**/_*.yml"
    files = glob.glob(pattern, recursive=True)
    if not files:
        print("✅ No files found to cleanup.")
        return

    print(f"\n⚠️  Found {len(files)} files to delete.")
    try:
        confirm = input("🔴 DELETE? (type 'yes'): ")
        if confirm.lower().strip() != 'yes': return
    except KeyboardInterrupt:
        print("\nOperation cancelled.")
        return

    for f in files:
        try:
            os.remove(f)
        except OSError as e:
            print(f"❌ Failed to delete {f}: {e}")
    print("✅ Cleanup done.")


async def action_cleanup_db(db):
    print("\n⚠️  WARNING: This will delete 'doc_cache' and 'doc_cache_log' tables from the database.")
    print("   This action cannot be undone.")
    try:
        confirm = input("🔴 DROP TABLES? (type 'yes'): ")
        if confirm.lower().strip() != 'yes': return
    except KeyboardInterrupt:
        print("\nOperation cancelled.")
        return

    print("🗑️  Dropping tables...")
    try:
        if db.type == 'postgres':
            async with db.pool.acquire() as conn:
                await conn.execute("DROP TABLE IF EXISTS doc_cache")
                await conn.execute("DROP TABLE IF EXISTS doc_cache_log")
        else:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, lambda: db.conn.execute("DROP TABLE IF EXISTS doc_cache"))
            await loop.run_in_executor(None, lambda: db.conn.execute("DROP TABLE IF EXISTS doc_cache_log"))
        print("✅ Tables dropped.")
    except Exception as e:
        print(f"❌ Failed to drop tables: {e}")


def action_run_osmosis():
    print("\n🚀 Running dbt-osmosis yaml refactor...")

    from shutil import which
    if which('dbt-osmosis') is None:
        print("❌ Error: 'dbt-osmosis' executable not found in PATH.")
        print("   Run: pip install dbt-osmosis")
        return

    try:
        subprocess.run(["dbt-osmosis", "yaml", "refactor"], check=True)
    except subprocess.CalledProcessError:
        print("❌ Error: dbt-osmosis returned non-zero exit code. Check your dbt project validity.")
    except Exception as e:
        print(f"❌ Unexpected error running dbt-osmosis: {e}")


async def process_single_yaml_file(file_path, db, use_ai, show_prompt):
    if "dbt_project.yml" in file_path or "dbt-autodoc.yml" in file_path: return

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.load(f)
    except (YAMLError, UnicodeDecodeError) as e:
        print(f"❌ SKIPPING broken file {os.path.basename(file_path)}: {e}")
        return
    except Exception as e:
        print(f"❌ Error reading {os.path.basename(file_path)}: {e}")
        return

    if not data or 'models' not in data: return

    changed = False
    
    # We need to collect tasks for this file
    tasks = []
    
    # Structure to hold context for applying results back
    # List of (model_index, column_index, coroutine)
    
    try:
        for m_idx, model_node in enumerate(data['models']):
            m_name = model_node.get('name')
            if not m_name: continue

            # Context extraction - we await here because it's a DB read needed for context
            table_desc_context = await db.get(m_name, TABLE_MARKER)
            if table_desc_context and AI_TAG in table_desc_context:
                table_desc_context = table_desc_context.replace(AI_TAG, "").strip()

            for c_idx, col in enumerate(model_node.get('columns', [])):
                c_name = col.get('name')
                curr_desc = col.get('description')
                
                # Create task
                coro = resolve_description(
                    curr_desc, m_name, c_name, db, use_ai,
                    is_table=False,
                    table_context=table_desc_context,
                    sql_content=None,
                    show_prompt=show_prompt
                )
                tasks.append((m_idx, c_idx, coro))
    except Exception as e:
        print(f"❌ Error setting up tasks for {os.path.basename(file_path)}: {e}")
        return

    if not tasks:
        return

    # Run all tasks for this file
    # Note: We process one file at a time in parallel internally, or we could parallelize across files.
    # Parallelizing across columns within a file is good.
    results = await asyncio.gather(*(t[2] for t in tasks))
    
    for i, res in enumerate(results):
        m_idx, c_idx, _ = tasks[i]
        
        # Apply result
        model_node = data['models'][m_idx]
        col = model_node['columns'][c_idx]
        curr_desc = col.get('description')
        
        if res and res != curr_desc:
            col['description'] = DoubleQuotedScalarString(res)
            changed = True

    if changed:
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(data, f)
        except Exception as e:
            print(f"❌ Failed to write back to {file_path}: {e}")

async def action_process_yaml_columns(db, use_ai=False, show_prompt=False):
    print(f"\n📂 Processing YAML Columns (AI={use_ai})...")
    yml_files = glob.glob(os.path.join(DBT_MODELS_DIR, "**/_*.yml"), recursive=True)

    if not yml_files:
        print(f"⚠️  No _*.yml files found in {DBT_MODELS_DIR}")
        return

    # Process files concurrently? 
    # Yes, but we need to be careful about file handles? Python handles are fine.
    # Limit concurrency to avoid too many open files or too much DB pressure?
    # Semaphore will limit AI calls. DB pool handles DB pressure.
    
    file_tasks = [process_single_yaml_file(f, db, use_ai, show_prompt) for f in yml_files]
    await asyncio.gather(*file_tasks)


async def process_single_sql_file(file_path, db, use_ai, show_prompt):
    m_name = os.path.splitext(os.path.basename(file_path))[0]

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"❌ SKIPPING reading {m_name}: {e}")
        return

    curr_desc = DbtConfigManipulator.extract_description(content)

    new_desc = await resolve_description(
        curr_desc, m_name, TABLE_MARKER, db, use_ai,
        is_table=True,
        table_context=None,
        sql_content=content,
        show_prompt=show_prompt
    )

    if new_desc and new_desc != curr_desc:
        new_content = DbtConfigManipulator.update_or_create(content, new_desc)
        if new_content != content:
            print(f"📝 Updating SQL: {m_name}")
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
            except Exception as e:
                print(f"❌ Failed to write SQL {m_name}: {e}")

async def action_process_sql_configs(db, use_ai=False, show_prompt=False):
    print(f"\n📄 Processing SQL Model Configs (AI={use_ai})...")
    sql_files = glob.glob(os.path.join(DBT_MODELS_DIR, "**/*.sql"), recursive=True)

    if not sql_files:
        print(f"⚠️  No .sql files found in {DBT_MODELS_DIR}")
        return

    tasks = [process_single_sql_file(f, db, use_ai, show_prompt) for f in sql_files]
    await asyncio.gather(*tasks)


def validate_dbt_project():
    if not os.path.exists("dbt_project.yml"):
        print("❌ Error: 'dbt_project.yml' not found.")
        print("   Please run this script from the root of your dbt project.")
        sys.exit(1)

    try:
        with open("dbt_project.yml", "r", encoding='utf-8') as f:
            project_data = yaml.load(f)
            if "+dbt-osmosis" not in str(project_data):
                print("❌ Missing '+dbt-osmosis' in dbt_project.yml")
                print("   Please configure dbt-osmosis before running.")
                sys.exit(1)
    except Exception as e:
        print(f"❌ Error reading dbt_project.yml: {e}")
        sys.exit(1)


# --- 7. MAIN ---

async def async_main():
    global model, concurrency_sem
    
    # Load environment variables from .env file
    load_dotenv()

    # --- HELP & EXAMPLES ---
    example_text = """
     EXAMPLES:
     
     dbt-autodoc --generate-docs-config-ai --generate-docs-yml-ai 
     dbt-autodoc --generate-docs-config-ai --gemini-api-key="AIzaSy..."
     dbt-autodoc --generate-docs-config-ai --show-prompt
     dbt-autodoc --cleanup-yml
    """

    parser = argparse.ArgumentParser(
        description="Automated DBT Documentation Generator using Google Gemini AI (Async)",
        epilog=example_text,
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument("--cleanup-yml", action="store_true", help="Delete _*.yml files")
    parser.add_argument("--cleanup-db", action="store_true", help="Drop database tables (doc_cache, doc_cache_log)")
    
    parser.add_argument("--generate-docs-yml", action="store_true", help="Run dbt-osmosis and sync YAML structure (No AI). Saves manual edits to DB.")
    parser.add_argument("--generate-docs-yml-ai", action="store_true", help="Run dbt-osmosis, sync YAML, and AI-generate column descriptions.")
    parser.add_argument("--generate-docs-config", action="store_true", help="Sync SQL config blocks (No AI). Saves manual edits to DB.")
    parser.add_argument("--generate-docs-config-ai", action="store_true", help="Sync SQL config blocks and AI-generate table descriptions.")

    parser.add_argument("--show-prompt", action="store_true", help="Print the prompt sent to AI for debugging")
    parser.add_argument("--gemini-api-key", type=str, help="Google Gemini API Key (overrides env var)")
    
    parser.add_argument("--concurrency", type=int, default=1, help="Max concurrent AI/DB requests (default: 1)")

    try:
        args = parser.parse_args()
    except SystemExit:
        return

    # --- INITIALIZE AI MODEL ---
    api_key = args.gemini_api_key or os.getenv('GEMINI_API_KEY') or CFG.get('gemini_api_key')
    use_ai = args.generate_docs_yml_ai or args.generate_docs_config_ai

    if api_key:
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        except Exception as e:
            print(f"⚠️  Failed to initialize Gemini: {e}")
            if use_ai:
                sys.exit(1)
    elif use_ai:
        print("❌ Error: AI generation requested but no API Key found.")
        print("   Provide key via --gemini-api-key or GEMINI_API_KEY env var.")
        sys.exit(1)

    # Initialize semaphore
    concurrency = args.concurrency or CFG.get('concurrency') or 10
    concurrency_sem = asyncio.Semaphore(concurrency)

    # --- VALIDATION ---
    if not args.cleanup_yml and not args.cleanup_db:
        validate_dbt_project()

    # --- CLEANUP MODE ---
    if args.cleanup_yml:
        action_cleanup()
        return

    # --- RUNTIME ---
    # We need project info for DB connection (schema) even if just cleaning up DB
    # But if cleanup_db is used, maybe we don't enforce dbt_project.yml existence?
    # However, DatabaseAdapter needs it. We can default to unknown if missing.
    
    project_info = get_dbt_project_info()
    
    db = DatabaseAdapter(project_info)
    await db.connect()
    
    if args.cleanup_db:
        await action_cleanup_db(db)
        await db.close()
        return

    await db.init_table()

    try:
        if args.generate_docs_yml or args.generate_docs_yml_ai:
            action_run_osmosis()
            await action_process_yaml_columns(db, use_ai=args.generate_docs_yml_ai, show_prompt=args.show_prompt)

        if args.generate_docs_config or args.generate_docs_config_ai:
            await action_process_sql_configs(db, use_ai=args.generate_docs_config_ai, show_prompt=args.show_prompt)

    except KeyboardInterrupt:
        print("\n🔴 Script interrupted by user. Exiting...")
    except Exception as e:
        print(f"\n❌ Unexpected crash: {e}")
    finally:
        await db.close()
        print("\n✨ Operation Complete.")

def main():
    asyncio.run(async_main())

if __name__ == "__main__":
    main()
