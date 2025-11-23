# DBT Autodoc Documentation

`dbt-autodoc` is an automated tool to generate and manage documentation for your dbt models using Google Gemini AI. It integrates with `dbt-osmosis` to synchronize your YAML files and ensures that your documentation is consistent, version-controlled, and easily maintainable.

## 🚀 Features

-   **Automated Generation:** Uses Google Gemini AI to generate technical descriptions for tables and columns.
-   **YAML Synchronization:** Keeps your `schema.yml` files in sync with your dbt models using `dbt-osmosis`.
-   **Caching & History:** Stores descriptions in a database (`duckdb` or `postgres`) to prevent regenerating existing documentation and tracks changes over time.
-   **User Tracking:** Logs who made changes to the documentation (based on environment variables or system user).
-   **Smart Updates:** Respects human-written documentation and allows forcing re-generation via tags.

## 🛠️ Setup

1.  **Install:**
    ```bash
    pip install dbt-autodoc
    # OR if you use Postgres:
    pip install dbt-autodoc[postgres]
    ```

2.  **Configuration:**
    When you run `dbt-autodoc` for the first time, it will automatically generate a `dbt-autodoc.yml` configuration file in your project root.
    
    **Important:** You should edit this file to provide context about your company (`company_context`), which significantly improves the AI's ability to generate accurate descriptions.
    
    **Supported AI:** Currently, this tool supports **Google Gemini** models (e.g., `gemini-2.5-flash`).

3.  **Environment Variables (Optional):**
    You can provide keys via environment variables (e.g. in a `.env` file) OR pass them as command-line arguments.
    ```env
    GEMINI_API_KEY=your_api_key_here
    POSTGRES_URL=postgresql://user:pass@host:port/db (if using postgres)
    DBT_USER=your_username (optional, for tracking)
    ```
    *Note: The tool attempts to load `.env` automatically. If that fails, it has a fallback to manually parse `POSTGRES_URL` from the `.env` file directly.*

## 📋 Recommended Workflow

For the best results, follow this workflow to build your documentation incrementally:

1.  **Initial Setup:**
    Run the tool once to generate `dbt-autodoc.yml`. Edit the file and fill in `company_context` with a detailed description of your business.

2.  **Step 1: Generate Table Descriptions (SQL)**
    Run the tool to generate descriptions for your models (tables/views) inside your SQL files.
    ```bash
    dbt-autodoc --generate-docs-config-ai
    ```
    *Why?* The AI uses the company context to describe what the table represents.

3.  **Step 2: Review & Refine Tables**
    Open your `.sql` files. Review the generated `{{ config(description=...) }}`.
    -   If it's good, leave it (or remove the `(ai_generated)` tag to lock it).
    -   If it's bad, edit it manually.
    -   **Run the tool again** to save your manual edits to the database "Source of Truth".

4.  **Step 3: Generate Column Descriptions (YAML)**
    Once table descriptions are solid, generate the column descriptions.
    ```bash
    dbt-autodoc --generate-docs-yml-ai
    ```
    *Why?* The AI now uses *both* the company context AND the specific table description to generate highly accurate column definitions.

5.  **Step 4: Review & Refine Columns**
    Check the generated `schema.yml` (or `_*.yml`) files.
    -   Refine definitions where necessary.
    -   Rerun `dbt-autodoc --generate-docs-yml` to sync your manual changes to the database.

6.  **Fast Track (Optional):**
    Once you are comfortable with the tool, you can run both generations at once. The tool will automatically prioritize tables first, then columns.
    ```bash
    dbt-autodoc --generate-docs-config-ai --generate-docs-yml-ai
    ```

## 🗄️ Database Selection: DuckDB vs Postgres

-   **DuckDB (`db_type: duckdb`)**:
    -   **Best for:** Individual developers, local testing, or single-user projects.
    -   **Pros:** Zero setup, fast, simple file-based database (`docs_backup.duckdb`).
    -   **Cons:** Cannot be easily shared concurrently between team members.

-   **Postgres (`db_type: postgres`)**:
    -   **Best for:** Production environments and Teams.
    -   **Pros:** Centralized "Source of Truth". Multiple developers can run the tool and share the same cache/history. If one developer documents a model, others get it automatically without regenerating.
    -   **Cons:** Requires a running Postgres instance.

## 📖 Usage & Arguments

Run the tool from the command line:

```bash
dbt-autodoc [ARGUMENTS]
```

> **Recommended:** Run `dbt run` (or `dbt compile`) before running this tool to ensure your project manifest is up to date.

### Available Arguments

| Argument | Description |
| :--- | :--- |
| `--generate-docs-yml` | **Sync Structure Only.** Runs `dbt-osmosis` to update YAML files with new columns/models. **Saves manual edits** to the database. Use this to sync files without AI. |
| `--generate-docs-yml-ai` | **Sync & Generate Columns.** Runs `dbt-osmosis`, then scans `_*.yml` files. If a column description is missing, calls AI to generate it. |
| `--generate-docs-config` | **Sync SQL Configs.** Updates SQL files. Read-only mode for descriptions (doesn't generate new ones). **Saves manual edits** to the database. |
| `--generate-docs-config-ai` | **Generate Table Descriptions.** Scans `.sql` model files. If a table description is missing in the `{{ config() }}` block, calls AI to generate it. |
| `--show-prompt` | **Debug Mode.** Prints the exact prompt sent to the AI without saving the result. Useful for testing prompt engineering. |
| `--cleanup-yml` | **Cleanup YAML.** Deletes temporary `_*.yml` files generated by osmosis if needed. |
| `--cleanup-db` | **Cleanup Database.** Drops the `doc_cache` and `doc_cache_log` tables from the database. Useful for resetting the schema or cache. **Irreversible.** |
| `--gemini-api-key` | Overrides the API key from environment variables. |
| `--concurrency` | Sets the maximum number of concurrent AI/DB requests (default: 10). Can also be set in `dbt-autodoc.yml`. |

## 🧠 How It Works

The tool follows a strict logic flow to determine whether to keep, update, or generate a description.

### 1. Description Resolution Logic

For every Column (in YAML) or Table (in SQL), the script follows a strict hierarchy to decide what to do. The goal is to **protect your manual work** while automating the rest.

1.  **Human Written (Highest Priority):**
    -   **Definition:** Any description that **does not** contain the `(ai_generated)` tag.
    -   **Behavior:** The script assumes you wrote this manually and that it is the "Source of Truth".
    -   **Action:** It **will NOT use AI** to generate another response. It **will NOT overwrite** your text. It effectively "locks" the description. It also saves this description to the database so that if you accidentally delete it later, it can be restored.

2.  **Existing AI:**
    -   **Definition:** A description containing the `(ai_generated)` tag.
    -   **Behavior:** The script considers this valid but "owned" by the machine.
    -   **Action:** It preserves the existing AI generation.
    -   **How to Regenerate:** If you want to regenerate an AI description, simply **delete it** from the file and run the script again with an AI flag (`--generate-docs-yml-ai` or `--generate-docs-config-ai`).

3.  **Cache Restore:**
    -   **Behavior:**
        -   If running **WITHOUT AI** (`--generate-docs-yml` or `--generate-docs-config`): The script attempts to restore missing descriptions from the `doc_cache` database. This protects against accidental deletion.
        -   If running **WITH AI** (`--generate-docs-yml-ai` or `--generate-docs-config-ai`): The script assumes a missing description means you want to generate a new one, so it **skips** the cache restore (unless the cached version was human-written).

4.  **Generate AI (Lowest Priority):**
    -   **Definition:** No description in file, and not restored from cache.
    -   **Action:** Calls Google Gemini to generate a new description.

#### 🔎 Examples

**Example 1: Protecting Manual Work (Human Override)**
-   **Scenario:** The AI previously generated: `"Flag indicating if user is active (ai_generated)"`.
-   **Your Action:** You decide this is too vague. You manually edit the YAML file to: `"Flag for users who have logged in within the last 30 days."` (Note: You removed the tag).
-   **Result:** On the next run, the tool sees no tag. It marks it as **Human Written**. It updates the database with your new definition but **will not** call the AI or overwrite your text. Your manual definition is now safe.

**Example 2: Forcing an AI Update (Regenerate)**
-   **Scenario:** The file has a description: `"Total value of orders (ai_generated)"` which you think is wrong.
-   **Your Action:** You delete the description line (or make it empty).
-   **Result:** Run `dbt-autodoc --generate-docs-yml-ai`. The tool sees the description is missing. It ignores the old cached AI value and calls Gemini to generate a fresh one.
-   **Final Output:** `"Sum of gross merchandise value for completed orders (ai_generated)"`.

**Example 3: Restoring Lost Documentation**
-   **Scenario:** You run a command that accidentally wipes descriptions, and you want them back without using AI.
-   **Action:** Run `dbt-autodoc --generate-docs-yml` (no AI).
-   **Result:** The tool checks `doc_cache` and restores your last known descriptions.

### 2. Special Tags

-   **(ai_generated)**: Automatically appended to all AI-generated descriptions. Identifies content that can be updated by the script.

### 3. Database & Caching

The tool maintains two tables in your database (`duckdb` local file or `postgres`):

#### `doc_cache`
Stores the *current* active description for every model and column.
-   **Purpose:** Prevents regenerating documentation for unchanged models (saves money/time) and serves as a backup.
-   **Columns:**
    -   `dbt_project_name`
    -   `dbt_profile_name`
    -   `model_name`
    -   `column_name`
    -   `description`
    -   `user_name`
    -   `is_human`: Boolean flag indicating if the description was manually written (True) or AI-generated (False).
    -   `updated_at`

#### `doc_cache_log`
An audit log of all changes made to descriptions.
-   **Purpose:** Tracks who changed what and when. Useful for debugging or rolling back.
-   **Trigger:** Written to whenever a description changes (e.g., `Old Value` -> `New Value`).
-   **Columns:**
    -   `dbt_project_name`
    -   `dbt_profile_name`
    -   `model_name`
    -   `column_name`
    -   `old_description`
    -   `new_description`
    -   `user_name`
    -   `is_human`: Boolean flag representing the status of the *new* description.
    -   `changed_at`

### 4. User Tracking

The tool tracks which user is running the script to populate the `user_name` field in the logs. It resolves the user in this order:
1.  `DBT_USER` environment variable.
2.  `USER` environment variable.
3.  `USERNAME` environment variable.
4.  System logged-in user (`getpass.getuser()`).
5.  Fallback to `'unknown'`.

## 📝 Best Practices

1.  **Run Structure Sync First:**
    Run `dbt-autodoc --generate-docs-yml` frequently to keep your YAML files consistent with your SQL models without calling AI.
2.  **Review AI Changes:**
    AI descriptions include the `(ai_generated)` tag. You can leave them as is, or edit them. If you remove the tag, the script will treat them as human-written and protect them from future updates.
3.  **Regenerate Poor AI Descriptions:**
    If an AI description is poor, simply delete it (from the YAML or SQL file) and run with `--generate-docs-yml-ai` (for columns) or `--generate-docs-config-ai` (for tables) to generate a new one.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
