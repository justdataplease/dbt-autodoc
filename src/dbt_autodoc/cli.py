import os
import glob
import time
import getpass
import subprocess
import argparse
import re
import sys
import threading
import concurrent.futures
import json
from functools import wraps
from dotenv import load_dotenv, dotenv_values, find_dotenv

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
        from ruamel.yaml.comments import CommentedMap
    except ImportError:
        missing.append("ruamel.yaml")
    
    # Check for psycopg2 if postgres is used
    try:
        import psycopg2
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
from ruamel.yaml.comments import CommentedMap, CommentedSeq


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

def get_env_var(var_name, default=None):
    """
    Retrieves environment variable with robust fallback to manual .env parsing.
    """
    val = os.getenv(var_name)
    if val:
        return val
    
    # Fallback: Try manual parsing if .env exists
    if os.path.exists(".env"):
        try:
            env_config = dotenv_values(".env")
            if var_name in env_config:
                val = env_config[var_name]
                print(f"Loaded {var_name} from .env manually.")
                return val
        except Exception as e:
            print(f"Manual .env parse failed for {var_name}: {e}")
            
    return default

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
manifest_data = None # Cache for manifest

# --- 3. DATABASE ADAPTER ---

def get_current_user():
    return os.getenv('DBT_USER') or os.getenv('USER') or os.getenv('USERNAME') or getpass.getuser() or 'unknown'


class DatabaseAdapter:
    def __init__(self, project_info=None):
        self.conn = None # For DuckDB
        self.pg_conn = None # For Postgres (single connection or pool? simple conn for now, threading safe?)
        # psycopg2 connections are thread safe, cursors are not shared across threads usually
        self.type = DB_TYPE
        self.project_name = project_info.get("name", "unknown") if project_info else "unknown"
        self.profile_name = project_info.get("profile", "unknown") if project_info else "unknown"
        self._lock = threading.Lock() # Lock for DuckDB or generic safety if needed

    def connect(self):
        try:
            if self.type == 'postgres':
                if not psycopg2:
                    print("❌ DB Type is 'postgres' but 'psycopg2' is not installed.")
                    print("   pip install psycopg2-binary")
                    sys.exit(1)

                postgres_url = get_env_var('POSTGRES_URL')

                if not postgres_url:
                    if os.path.exists(".env"):
                        raise ValueError("POSTGRES_URL not found in environment, but .env file exists. Check variable name and format.")
                    else:
                        raise ValueError("POSTGRES_URL environment variable is missing and no .env file found.")
                
                self.pg_conn = psycopg2.connect(postgres_url)
                self.pg_conn.autocommit = True
            else:
                self.conn = duckdb.connect(DUCKDB_PATH)
        except Exception as e:
            print(f"❌ CRITICAL: Could not connect to database ({self.type}).")
            if "IO Error" in str(e) and self.type == 'duckdb':
                print("   (Hint: Is another process/DBeaver holding the .duckdb file open?)")
            print(f"   Error details: {e}")
            sys.exit(1)

    def init_table(self):
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
                    with self.pg_conn.cursor() as cur:
                        cur.execute(q_cache_pg)
                        cur.execute(q_log_pg)
                else:
                    self.conn.execute(q_cache_duck)
                    self.conn.execute(q_log_duck)
                
                if self.migrate_schema():
                    print("♻️  Schema mismatch detected & handled. Re-initializing tables...")
                    continue # Retry create after drop
                break # Done if no migration needed or handled
            except Exception as e:
                print(f"❌ Error initializing database table: {e}")
                print("   If you have an old database schema, you might need to run with --cleanup-db.")
                sys.exit(1)

    def migrate_schema(self):
        # Simplified migration: If critical columns missing, DROP tables to reset.
        try:
            tables = ["doc_cache", "doc_cache_log"]
            required_columns = ["dbt_project_name", "dbt_profile_name", "user_name", "is_human"]
            
            needs_reset = False

            for table in tables:
                existing_cols = []
                if self.type == 'postgres':
                    with self.pg_conn.cursor() as cur:
                        cur.execute(f"SELECT column_name FROM information_schema.columns WHERE table_name = '{table}'")
                        rows = cur.fetchall()
                        existing_cols = [row[0] for row in rows]
                else:
                    try:
                        info = self.conn.execute(f"PRAGMA table_info('{table}')").fetchall()
                        existing_cols = [col[1] for col in info]
                    except:
                        existing_cols = []
                
                if not existing_cols: continue # Table doesn't exist yet (handled by init)

                for req in required_columns:
                    if req not in existing_cols:
                        print(f"⚠️  Schema mismatch in {table}: Missing {req}. Resetting table (Dev Mode)...")
                        needs_reset = True
                        break
                if needs_reset: break

            if needs_reset:
                action_cleanup_db(self)
                return True # Signal to re-init

            return False

        except Exception as e:
            print(f"⚠️  Migration Check Warning: {e}")
            return False

    def get(self, model, col):
        try:
            if self.type == 'duckdb':
                q = "SELECT description FROM doc_cache WHERE dbt_project_name = ? AND dbt_profile_name = ? AND model_name = ? AND column_name = ?"
                params = (self.project_name, self.profile_name, model, col)
                # Use cursor for thread safety
                res = self.conn.cursor().execute(q, params).fetchone()
                return res[0] if res else None
            else:
                q = "SELECT description FROM doc_cache WHERE dbt_project_name = %s AND dbt_profile_name = %s AND model_name = %s AND column_name = %s"
                with self.pg_conn.cursor() as cur:
                    cur.execute(q, (self.project_name, self.profile_name, model, col))
                    res = cur.fetchone()
                    return res[0] if res else None
        except Exception as e:
            print(f"⚠️  DB Read Error ({model}.{col}): {e}")
            return None

    def save(self, model, col, description):
        if not description: return
        try:
            clean_desc = str(description).strip('"')
            old_desc = self.get(model, col)
            user = get_current_user()
            is_human = AI_TAG not in clean_desc
            
            # Only log if there is a change
            if old_desc != clean_desc:
                self.log_change(model, col, old_desc, clean_desc, user, is_human)

            if self.type == 'postgres':
                q = """
                    INSERT INTO doc_cache (dbt_project_name, dbt_profile_name, model_name, column_name, description, user_name, is_human, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                    ON CONFLICT (dbt_project_name, dbt_profile_name, model_name, column_name)
                    DO UPDATE SET description = EXCLUDED.description, user_name = EXCLUDED.user_name, is_human = EXCLUDED.is_human, updated_at = CURRENT_TIMESTAMP
                    """
                with self.pg_conn.cursor() as cur:
                    cur.execute(q, (self.project_name, self.profile_name, model, col, clean_desc, user, is_human))
            else:
                q = """
                    INSERT OR REPLACE INTO doc_cache (dbt_project_name, dbt_profile_name, model_name, column_name, description, user_name, is_human, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """
                # Use cursor for thread safety
                self.conn.cursor().execute(q, (self.project_name, self.profile_name, model, col, clean_desc, user, is_human))
        except Exception as e:
            print(f"⚠️  DB Save Error ({model}.{col}): {e}")

    def log_change(self, model, col, old_desc, new_desc, user, is_human):
        try:
            if self.type == 'postgres':
                q = """
                    INSERT INTO doc_cache_log (dbt_project_name, dbt_profile_name, model_name, column_name, old_description, new_description, user_name, is_human, changed_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                    """
                with self.pg_conn.cursor() as cur:
                    cur.execute(q, (self.project_name, self.profile_name, model, col, old_desc, new_desc, user, is_human))
            else:
                q = """
                    INSERT INTO doc_cache_log (dbt_project_name, dbt_profile_name, model_name, column_name, old_description, new_description, user_name, is_human, changed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """
                self.conn.cursor().execute(q, (self.project_name, self.profile_name, model, col, old_desc, new_desc, user, is_human))
        except Exception as e:
            print(f"⚠️  DB Log Error ({model}.{col}): {e}")

    def close(self):
        if self.type == 'postgres' and self.pg_conn:
            try:
                self.pg_conn.close()
            except:
                pass
        elif self.type == 'duckdb' and self.conn:
            try:
                self.conn.close()
            except:
                pass


# --- 4. MANIFEST / UPSTREAM EXTENSION ---
def load_manifest(manifest_path="target/manifest.json"):
    """Loads dbt manifest.json to understand the DAG."""
    global manifest_data
    if manifest_data:
        return manifest_data

    if not os.path.exists(manifest_path):
        print(f"⚠️  Manifest file not found at {manifest_path}.")
        print("   Upstream context (dependencies) will be unavailable.")
        print("   💡 Tip: Run 'dbt compile' or 'dbt docs generate' to create the manifest file.")
        return None
    
    try:
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest_data = json.load(f)
            return manifest_data
    except Exception as e:
        print(f"❌ Failed to load manifest: {e}")
        return None

def get_upstream_models(model_name, manifest=None, project_name=None):
    """
    Finds direct upstream parents (depends_on) for a given model.
    Returns a list of parent model names.
    """
    if not manifest:
        manifest = load_manifest()
    
    if not manifest or 'nodes' not in manifest:
        return []

    # Find the node for this model
    # Nodes in manifest are keyed like 'model.project_name.model_name'
    target_node_id = None
    target_node = None
    
    # Strategy 1: strict match if project_name is known
    if project_name:
        # Standard dbt node ID format: model.<project_name>.<model_name>
        # Note: Depending on dbt version/config, it might vary, but this is standard.
        candidate_id = f"model.{project_name}.{model_name}"
        if candidate_id in manifest['nodes']:
            target_node = manifest['nodes'][candidate_id]

    # Strategy 2: Search by name if not found yet
    if not target_node:
        for node_id, node in manifest['nodes'].items():
            if node.get('name') == model_name and node.get('resource_type') == 'model':
                # Optional: If we have a project name, ensure it matches
                if project_name and node.get('package_name') != project_name:
                    continue
                target_node = node
                break
            
    if not target_node:
        return []
    
    # Get depends_on nodes
    depends_on = target_node.get('depends_on', {}).get('nodes', [])
    
    # Filter for models only (exclude sources/macros if desired, or keep them)
    # We usually want models or sources to give context.
    
    upstream_info = []
    
    for parent_id in depends_on:
        parent_node = manifest['nodes'].get(parent_id)
        if not parent_node:
            # Maybe it's a source?
            if parent_id in manifest.get('sources', {}):
                 parent_node = manifest['sources'][parent_id]
        
        if parent_node:
            name = parent_node.get('name')
            desc = parent_node.get('description', '')
            resource_type = parent_node.get('resource_type')
            upstream_info.append({
                "name": name,
                "type": resource_type,
                "description": desc
            })
            
    return upstream_info


# --- 5. AI HELPER ---

def ask_gemini(model_name, target_name, is_table=False, table_context=None, sql_content=None, show_prompt=False, project_name=None):
    if not model:
        return None

    entity_type = "Table" if is_table else "Column"
    context_block = f"\n    Parent Table Context: {table_context}\n" if (table_context and not is_table) else ""

    sql_block = ""
    if is_table and sql_content:
        safe_sql = sql_content[:15000]
        sql_block = f"\n    SQL Source Code:\n    ```sql\n{safe_sql}\n    ```\n"
    elif sql_content:
        # For columns, we also include the model SQL if available
        safe_sql = sql_content[:15000]
        sql_block = f"\n    Model SQL Source Code:\n    ```sql\n{safe_sql}\n    ```\n"
    
    # --- UPSTREAM CONTEXT ADDITION ---
    upstream_context_str = ""
    # Always include upstream context for both tables and columns
    upstream = get_upstream_models(model_name, project_name=project_name)
    if upstream:
        upstream_context_str = "\n    UPSTREAM MODELS (Dependencies):\n"
        for u in upstream:
            desc = u['description'] if u['description'] else "No description"
            upstream_context_str += f"    - {u['type']} {u['name']}: {desc}\n"
    
    prompt = f"""
    You are a Data Dictionary Editor. Your goal is to write technical, dry, and precise definitions.

    INPUT CONTEXT:
    - Business Context: {COMPANY_CONTEXT} (Use this to understand the logic, but DO NOT use the company name in the output).
    - Model Name: {model_name}
    - Type: {entity_type}
    - Object Name: {target_name}
    {context_block}
    {upstream_context_str}
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
        print(f"🤖 Asking AI for {model_name} -> {target_name}...")
        response = model.generate_content(prompt)
            
        if not response.text:
            print("⚠️  AI returned empty text (possibly safety filtered).")
            return None

        text = response.text.strip().strip('"').strip("'")
        return f"{text} {AI_TAG}"
    except Exception as e:
        print(f"⚠️  AI API Error: {e}")
        return None


# --- 6. CENTRAL LOGIC ---

def resolve_description(current_desc, model_name, col_name, db, use_ai, is_table=False, table_context=None, sql_content=None, show_prompt=False):
    current_desc_str = str(current_desc) if current_desc else ""

    # 1. Keep Human Written
    if current_desc_str and AI_TAG not in current_desc_str:
        db.save(model_name, col_name, current_desc_str)
        return current_desc_str

    # 2. Keep Existing AI
    if current_desc_str and AI_TAG in current_desc_str:
        db.save(model_name, col_name, current_desc_str)
        return current_desc_str

    # 3. Restore from DB
    cached_desc = db.get(model_name, col_name)
    if cached_desc:
        is_human_cached = AI_TAG not in cached_desc

        if is_human_cached:
            print(f"💾 Restored Human Description from DB: {model_name}.{col_name}")
            return cached_desc
        
        # Fix: Always restore AI description if found in DB (even if use_ai=True)
        # to prevent unnecessary re-generation.
        print(f"💾 Restored AI Description from DB: {model_name}.{col_name}")
        return cached_desc

    # 4. Ask AI
    if use_ai:
        ai_text = ask_gemini(model_name, col_name, is_table, table_context, sql_content, show_prompt, project_name=db.project_name)
        if ai_text:
            db.save(model_name, col_name, ai_text)
            print(f"✅ Saved AI Description for {model_name}.{col_name}")
            return ai_text

    return current_desc


# --- 7. ACTIONS ---

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


def action_cleanup_db(db):
    print("\n⚠️  WARNING: This will delete 'doc_cache' and 'doc_cache_log' tables from the database.")
    print("   This action cannot be undone. Make sure to backup your existing data not to lose it.")
    try:
        confirm = input("🔴 DROP TABLES? (type 'yes'): ")
        if confirm.lower().strip() != 'yes': return
    except KeyboardInterrupt:
        print("\nOperation cancelled.")
        return

    print("🗑️  Dropping tables...")
    try:
        if db.type == 'postgres':
            with db.pg_conn.cursor() as cur:
                cur.execute("DROP TABLE IF EXISTS doc_cache")
                cur.execute("DROP TABLE IF EXISTS doc_cache_log")
        else:
            db.conn.execute("DROP TABLE IF EXISTS doc_cache")
            db.conn.execute("DROP TABLE IF EXISTS doc_cache_log")
        print("✅ Tables dropped.")
    except Exception as e:
        print(f"❌ Failed to drop tables: {e}")


def action_run_osmosis(with_inheritance=False):
    print("\n🚀 Syncing YAML files...")

    from shutil import which
    if which('dbt-osmosis') is None:
        print("❌ Error: 'dbt-osmosis' executable not found in PATH.")
        print("   Run: pip install dbt-osmosis")
        return

    try:
        cmd = [
            "dbt-osmosis", "yaml", "refactor",
            "--auto-apply"
        ]
        if with_inheritance:
            cmd.extend(["--force-inherit-descriptions", "--use-unrendered-descriptions"])

        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        print("❌ Error: dbt-osmosis returned non-zero exit code. Check your dbt project validity.")
    except Exception as e:
        print(f"❌ Unexpected error running dbt-osmosis: {e}")


def find_model_sql_path(model_name, base_dir):
    # Try to find the SQL file for the model
    search_pattern = os.path.join(base_dir, "**", f"{model_name}.sql")
    files = glob.glob(search_pattern, recursive=True)
    return files[0] if files else None

def _reorder_map_keys(node: CommentedMap, preferred_order: list):
    if not isinstance(node, CommentedMap):
        return

    # Store original items with their comments
    items_with_comments = []
    for key in list(node.keys()):
        value = node.pop(key)
        # ruamel.yaml stores comments in various places. This is a best effort.
        # It's challenging to reliably extract *all* comment types associated with a key
        # without deep knowledge of ruamel.yaml's internal comment structure.
        # This approach tries to capture comments that might be 'before' the key or 'eol' (end-of-line).
        
        # Get comment before the key
        pre_comment = node.yaml_get_comment_before_after_key(key, _info=True)
        before_comment = pre_comment[0][2].split('\n') if pre_comment and pre_comment[0] and pre_comment[0][2] else []
        
        # Get end-of-line comment
        eol_comment = node.yaml_get_item_comment(key)
        
        items_with_comments.append({'key': key, 'value': value, 'before': before_comment, 'eol': eol_comment})
    
    # Sort items based on preferred_order, then alphabetically for remaining
    def sort_key_func(item):
        key = item['key']
        try:
            return (preferred_order.index(key), key)
        except ValueError:
            return (len(preferred_order), key) # Place unspecified keys at the end, then alpha

    items_with_comments.sort(key=sort_key_func)

    # Re-insert into the node, attempting to restore comments
    for item in items_with_comments:
        key = item['key']
        value = item['value']
        
        # Insert key-value pair
        node[key] = value

        # Re-attach comments
        if item['before']:
            # For multiline comments, ruamel.yaml expects a single string with newlines
            node.yaml_set_comment_before_after_key(key, before=''.join(item['before']))
        if item['eol']:
            node.yaml_set_comment_before_after_key(key, after=item['eol'])


def process_single_yaml_file(file_path, db, use_ai, show_prompt, executor, model_path_override=None, scope='both'):
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
    
    # Base directory for searching SQL files
    base_sql_dir = model_path_override if model_path_override else DBT_MODELS_DIR

    # Preferred order for model keys
    preferred_model_keys_order = ['name', 'description', 'access', 'config', 'columns']
    # Preferred order for column keys
    preferred_column_keys_order = ['name', 'description', 'data_type', 'tests', 'meta']


    # Iterate over models sequentially
    for m_idx, model_node in enumerate(data['models']):
        m_name = model_node.get('name')
        if not m_name: continue

        # Context extraction - sync
        table_desc_context = db.get(m_name, TABLE_MARKER)
        if table_desc_context and AI_TAG in table_desc_context:
            table_desc_context = table_desc_context.replace(AI_TAG, "").strip()

        # Get SQL content for context
        sql_content = None
        sql_file = find_model_sql_path(m_name, base_sql_dir)
        if sql_file:
            try:
                with open(sql_file, 'r', encoding='utf-8') as sf:
                    sql_content = sf.read()
            except Exception:
                pass # Ignore errors reading SQL file

        # Collect tasks for this model (columns)
        futures = {}
        
        # --- MODEL DESCRIPTION TASK ---
        if scope in ['tables', 'both']:
            model_desc_curr = model_node.get('description')
            
            future_model = executor.submit(
                resolve_description,
                model_desc_curr, m_name, TABLE_MARKER, db, use_ai,
                is_table=True,
                table_context=None,
                sql_content=sql_content,
                show_prompt=show_prompt
            )
            # Use -1 as column index to indicate Model Task
            futures[future_model] = (m_idx, -1)

        # --- COLUMN DESCRIPTION TASKS ---
        if scope in ['columns', 'both']:
            for c_idx, col in enumerate(model_node.get('columns', [])):
                c_name = col.get('name')
                curr_desc = col.get('description')
                
                # Submit task
                future = executor.submit(
                    resolve_description,
                    curr_desc, m_name, c_name, db, use_ai,
                    is_table=False,
                    table_context=table_desc_context,
                    sql_content=sql_content,
                    show_prompt=show_prompt
                )
                futures[future] = (m_idx, c_idx)

        # Wait for all tasks for this model to finish
        for future in concurrent.futures.as_completed(futures):
            _, c_idx = futures[future]
            try:
                res = future.result()
                
                if c_idx == -1: # Model Description
                    curr_desc = model_node.get('description')
                    if res and res != curr_desc:
                        # Ensure description is inserted after 'name'
                        if 'description' not in model_node and 'name' in model_node:
                            name_idx = list(model_node.keys()).index('name')
                            model_node.insert(name_idx + 1, 'description', DoubleQuotedScalarString(res))
                        else:
                            model_node['description'] = DoubleQuotedScalarString(res)
                        changed = True
                else: # Column Description
                    col = model_node['columns'][c_idx]
                    curr_desc = col.get('description')
                    
                    if res and res != curr_desc:
                        # Ensure description is inserted after 'name' in columns too
                        if 'description' not in col and 'name' in col:
                            name_idx = list(col.keys()).index('name')
                            col.insert(name_idx + 1, 'description', DoubleQuotedScalarString(res))
                        else:
                            col['description'] = DoubleQuotedScalarString(res)
                        changed = True
            except Exception as e:
                print(f"❌ Error processing in {m_name}: {e}")
        
        # After processing model and its columns, reorder keys
        _reorder_map_keys(model_node, preferred_model_keys_order)
        if 'columns' in model_node and isinstance(model_node['columns'], CommentedSeq):
            for col in model_node['columns']:
                if isinstance(col, CommentedMap):
                    _reorder_map_keys(col, preferred_column_keys_order)

    if changed:
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(data, f)
        except Exception as e:
            print(f"❌ Failed to write back to {file_path}: {e}")

def action_process_yaml_columns(db, use_ai=False, show_prompt=False, concurrency=10, model_path=None, scope='both'):
    print(f"\n📂 Processing YAML Models & Columns (AI={use_ai}, scope={scope})...")
    target_dir = model_path if model_path else DBT_MODELS_DIR
    yml_files = glob.glob(os.path.join(target_dir, "**/_*.yml"), recursive=True)

    if not yml_files:
        print(f"⚠️  No _*.yml files found in {target_dir}")
        return

    # Process files sequentially to respect "not multiple tables concurrent" at top level
    # Inside each file (if multiple tables, handled in process_single_yaml_file), we use executor for columns.
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        for f in yml_files:
            process_single_yaml_file(f, db, use_ai, show_prompt, executor, model_path_override=model_path, scope=scope)

def action_sort_yml(model_path=None):
    print("\n reorganizing keys and sorting...")
    target_dir = model_path if model_path else DBT_MODELS_DIR
    yml_files = glob.glob(os.path.join(target_dir, "**/_*.yml"), recursive=True)

    if not yml_files:
        print(f"⚠️  No _*.yml files found in {target_dir}")
        return

    # Preferred order for model keys
    preferred_model_keys_order = ['name', 'description', 'access', 'config', 'columns']
    # Preferred order for column keys
    preferred_column_keys_order = ['name', 'description', 'data_type', 'tests', 'meta']

    for file_path in yml_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.load(f)
        except (YAMLError, UnicodeDecodeError) as e:
            print(f"❌ SKIPPING broken file {os.path.basename(file_path)}: {e}")
            continue
        except Exception as e:
            print(f"❌ Error reading {os.path.basename(file_path)}: {e}")
            continue

        if not data or 'models' not in data: continue

        changed = False
        for model_node in data['models']:
            if isinstance(model_node, CommentedMap):
                _reorder_map_keys(model_node, preferred_model_keys_order)
                if 'columns' in model_node and isinstance(model_node['columns'], CommentedSeq):
                    for col in model_node['columns']:
                        if isinstance(col, CommentedMap):
                            _reorder_map_keys(col, preferred_column_keys_order)
                changed = True
        
        if changed:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    yaml.dump(data, f)
                print(f"✅ Reordered keys in {os.path.basename(file_path)}")
            except Exception as e:
                print(f"❌ Failed to write back to {file_path} after sorting: {e}")


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

def main():
    global model
    
    # Load environment variables from .env file (search up directories)
    load_dotenv(find_dotenv(usecwd=True))

    # --- HELP & EXAMPLES ---
    example_text = """
     EXAMPLES:
     
     dbt-autodoc --generate-docs-model-ai --generate-docs-model-columns-ai 
     dbt-autodoc --generate-docs-model-ai --gemini-api-key="AIzaSy..."
     dbt-autodoc --generate-docs-model-ai --show-prompt
     dbt-autodoc --cleanup-yml
    """

    parser = argparse.ArgumentParser(
        description="Automated DBT Documentation Generator using Google Gemini AI (Sync + Threads + Upstream Context)",
        epilog=example_text,
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument("--cleanup-yml", action="store_true", help="Delete _*.yml files generated by osmosis.")
    parser.add_argument("--cleanup-db", action="store_true", help="Drop database tables (doc_cache, doc_cache_log). useful for resetting the schema.")
    
    parser.add_argument("--generate-docs-model", action="store_true", help="Sync model descriptions in YAML from cache (No AI).")
    parser.add_argument("--generate-docs-model-ai", action="store_true", help="AI-generate model descriptions in YAML.")
    parser.add_argument("--generate-docs-model-columns", action="store_true", help="Sync column descriptions in YAML from cache (No AI).")
    parser.add_argument("--generate-docs-model-columns-ai", action="store_true", help="AI-generate column descriptions in YAML.")
    
    parser.add_argument("--regenerate-yml", action="store_true", help="Regenerate YAML files from dbt models (preserves descriptions).")
    parser.add_argument("--regenerate-yml-with-inheritance", action="store_true", help="Regenerate YAML files with description inheritance enabled.")
    parser.add_argument("--generate-docs", action="store_true", help="Run full documentation flow (Tables -> Sync -> Columns) WITHOUT AI.")
    parser.add_argument("--generate-docs-ai", action="store_true", help="Run full documentation flow (Tables -> Sync -> Columns) WITH AI.")

    parser.add_argument("--show-prompt", action="store_true", help="Print the prompt sent to AI for debugging")
    parser.add_argument("--gemini-api-key", type=str, help="Google Gemini API Key (overrides env var)")
    
    parser.add_argument("--concurrency", type=int, default=None, help="Max concurrent threads (default: 10).")
    parser.add_argument("--model-path", type=str, default=None, help="Specific directory to process (e.g. models/staging). Defaults to configured dbt_models_dir.")
    parser.add_argument("--sort-yml", action="store_true", help="Sort keys in YAML files (name, description, columns for models; name, description for columns).")

    try:
        args = parser.parse_args()
    except SystemExit:
        return

    # --- INITIALIZE AI MODEL ---
    api_key = args.gemini_api_key or get_env_var('GEMINI_API_KEY') or CFG.get('gemini_api_key')
    # Determine if AI is needed
    use_ai = args.generate_docs_model_ai or args.generate_docs_model_columns_ai or args.generate_docs_ai

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

    # Concurrency
    concurrency_val = args.concurrency
    if concurrency_val is None:
        concurrency_val = CFG.get('concurrency', 10)
    
    try:
        concurrency = int(concurrency_val)
    except (ValueError, TypeError):
        print(f"⚠️  Invalid concurrency value: {concurrency_val}. Using default 10.")
        concurrency = 10

    # --- VALIDATION ---
    if not args.cleanup_yml and not args.cleanup_db and not args.sort_yml:
        validate_dbt_project()

    # --- CLEANUP MODE ---
    if args.cleanup_yml:
        action_cleanup()
        return

    # --- SORT ONLY MODE ---
    if args.sort_yml:
        action_sort_yml(model_path=args.model_path)
        return


    # --- RUNTIME ---
    project_info = get_dbt_project_info()
    
    db = DatabaseAdapter(project_info)
    db.connect()
    
    if args.cleanup_db:
        action_cleanup_db(db)
        db.close()
        return

    db.init_table()

    # Attempt to load manifest once
    load_manifest()

    try:
        # New Combined Flows
        if args.regenerate_yml:
            action_run_osmosis(with_inheritance=False)

        elif args.regenerate_yml_with_inheritance:
            action_run_osmosis(with_inheritance=True)

        elif args.generate_docs:
            # Full flow NO AI
            # 1. Osmosis (Sync structure & inherit)
            action_run_osmosis(with_inheritance=False)
            # 2. Process YAMLs (Models & Columns with AI)
            action_process_yaml_columns(db, use_ai=False, show_prompt=args.show_prompt, concurrency=concurrency, model_path=args.model_path, scope='both')
            # 3. Osmosis (Final check/format)
            action_run_osmosis(with_inheritance=False)
            # 4. Process YAMLs (Sync again)
            action_process_yaml_columns(db, use_ai=False, show_prompt=args.show_prompt, concurrency=concurrency, model_path=args.model_path, scope='both')

        elif args.generate_docs_ai:
            # Full flow WITH AI
            # 1. Osmosis (Sync structure & inherit)
            action_run_osmosis(with_inheritance=False)
            # 2. Process YAMLs (Models & Columns with AI)
            action_process_yaml_columns(db, use_ai=True, show_prompt=args.show_prompt, concurrency=concurrency, model_path=args.model_path, scope='both')
            # 3. Osmosis (Final check/format)
            action_run_osmosis(with_inheritance=False)
            # 4. Process YAMLs (Sync again)
            action_process_yaml_columns(db, use_ai=False, show_prompt=args.show_prompt, concurrency=concurrency, model_path=args.model_path, scope='both')

        # Individual Flags (New behavior)
        else:
            if args.generate_docs_model or args.generate_docs_model_ai:
                action_run_osmosis(with_inheritance=False) # Ensure YAML structure is there
                action_process_yaml_columns(db, use_ai=args.generate_docs_model_ai, show_prompt=args.show_prompt, concurrency=concurrency, model_path=args.model_path, scope='tables')

            if args.generate_docs_model_columns or args.generate_docs_model_columns_ai:
                action_run_osmosis(with_inheritance=False) # Ensure YAML structure is there
                action_process_yaml_columns(db, use_ai=args.generate_docs_model_columns_ai, show_prompt=args.show_prompt, concurrency=concurrency, model_path=args.model_path, scope='columns')

    except KeyboardInterrupt:
        print("\n🔴 Script interrupted by user. Exiting...")
    except Exception as e:
        print(f"\n❌ Unexpected crash: {e}")
    finally:
        db.close()
        print("\n✨ Operation Complete.")

if __name__ == "__main__":
    main()
