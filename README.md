# DBT Autodoc Documentation

`dbt-autodoc` is the ultimate tool for **Automated Documentation** and **Logging** for your dbt projects. It combines the power of Google Gemini AI with a robust **Database Logging** system to ensure your documentation is always up-to-date, accurate, and auditable.

## 🌟 Why dbt-autodoc?

-   **🤖 Automatic AI Documentation:** Generate comprehensive descriptions for your tables and columns automatically.
-   **💾 Database Logging & History:** Every description is stored in a database (`duckdb` or `postgres`). This acts as a "Source of Truth" and provides a full history of changes.
-   **🔄 Full Synchronization:** Seamlessly integrates with `dbt-osmosis` to keep your YAML files in sync with your SQL models.
-   **🔒 Protect Manual Work:** Respects human-written documentation. If you write it, we lock it.
-   **👥 Team Ready:** Use Postgres to share documentation cache across your entire team.

## 🛠️ Setup

1.  **Install:**
    ```bash
    pip install dbt-autodoc
    ```

2.  **Configuration:**
    Run `dbt-autodoc --help` to generate `dbt-autodoc.yml`.
    **Important:** Edit `company_context` in this file to give the AI knowledge about your business logic.

3.  **Environment Variables:**
    ```env
    GEMINI_API_KEY=your_api_key_here
    POSTGRES_URL=postgresql://user:pass@host:port/db (optional)
    ```

## 📋 Recommended Workflow

For the best results, follow this step-by-step workflow to ensure accuracy and control:

1.  **Preparation:**
    Update your dbt project, generate the manifest, and context.
    ```bash
    dbt run && dbt docs generate
    # Edit dbt-autodoc.yml with company_context
    ```

2.  **Sync Structure (No AI):**
    Regenerate YAML files to match the SQL models. This ensures all new columns are present.
    ```bash
    dbt-autodoc --regenerate-yml
    ```

3.  **Generate Model Descriptions (YAML):**
    Generate AI descriptions for your models (tables/views).
    ```bash
    dbt-autodoc --generate-docs-model-ai --model-path models/staging
    ```

4.  **Manual Review (Important):**
    Open your YAML files. Review the structure and any existing descriptions. If you manually update a description here, it will be protected from AI overwrites in the next step.

5.  **Generate Model Column Descriptions (YAML):**
    Use AI to fill in the missing column descriptions.
    ```bash
    dbt-autodoc --generate-docs-model-columns-ai --model-path models/staging
    ```

6.  **Propagate & Save:**
    Run inheritance rules on the entire dbt project, then run the tool again to save the final state (including inherited descriptions) to the database.
    ```bash
    dbt-autodoc --regenerate-yml-with-inheritance
    dbt-autodoc --generate-docs-model-columns-ai --model-path models/staging
    ```

7.  **Next Layer:**
    Repeat steps 2-6 for `models/intermediate`, `models/marts`, etc.

## 🚀 Quick Start (Automated)

If you trust the process and just want to run everything at once:

```bash
dbt-autodoc --generate-docs-ai
```

## 🧠 How the AI Works

When generating a description for a column or table, the AI considers multiple inputs to produce the most accurate result:

1.  **Company Context:** The high-level business logic defined in your config.
2.  **Model SQL:** The actual code of the model being documented.
3.  **Existing Descriptions:** Any existing documentation or comments in the file.
4.  **Upstream Logic:** (Implicitly via Osmosis inheritance) Context from upstream models.

It synthesizes all these inputs to write a concise, technical description.

## 📖 Arguments Reference

| Argument | Description |
| :--- | :--- |
| `--regenerate-yml` | **Structure Only.** Regenerate YAML files from dbt models. Does not sync to DB or call AI. |
| `--regenerate-yml-with-inheritance` | **Structure + Inheritance.** Regenerate YAML files with inheritance enabled. Use this to propagate descriptions from upstream models. |
| `--model-path` | Restrict processing to a specific directory (e.g. `models/staging`). |
| `--generate-docs-model-ai` | Generate model descriptions in `.yml` files using AI. |
| `--generate-docs-model-columns-ai` | Generate column descriptions in `.yml` files using AI. |
| `--generate-docs-model` | Sync model descriptions in `.yml` files from cache (no AI). |
| `--generate-docs-model-columns` | Sync column descriptions in `.yml` files from cache (no AI). |
| `--generate-docs-ai` | **🔥 Full Auto.** Runs the complete workflow: Model generation, YAML sync, and Column generation using AI. |
| `--generate-docs` | **🔄 Full Sync.** Runs the complete workflow using only the database cache (no AI). |
| `--cleanup-db` | **Reset Database.** Wipes the description cache and history. |
| `--concurrency` | Max threads for AI/DB requests (default: 10). |

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Attribution

Brought to you by [JustDataPlease](https://justdataplease.com/agency/).
