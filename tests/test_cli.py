import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import sys
import os

# We need to import the module. 
# Since the module has global execution that relies on config file presence,
# we assume dbt-autodoc.yml exists in the CWD or we mock it.
# But mocking builtins.open or os.path.exists during import is hard with standard pytest import mechanisms.
# For now, we assume the environment is set up correctly (dependencies installed, config present).

from dbt_autodoc import cli

class TestDbtConfigManipulator:
    def test_extract_description_simple(self):
        sql = """
        {{ config(
            materialized='table',
            description="This is a test table"
        ) }}
        SELECT * FROM source
        """
        desc = cli.DbtConfigManipulator.extract_description(sql)
        assert desc == "This is a test table"

    def test_extract_description_no_description(self):
        sql = """
        {{ config(
            materialized='table'
        ) }}
        SELECT * FROM source
        """
        desc = cli.DbtConfigManipulator.extract_description(sql)
        assert desc is None

    def test_update_or_create_update_existing(self):
        sql = """
        {{ config(
            description="Old description"
        ) }}
        """
        new_desc = "New description"
        updated_sql = cli.DbtConfigManipulator.update_or_create(sql, new_desc)
        assert 'description="New description"' in updated_sql
        assert "Old description" not in updated_sql

    def test_update_or_create_add_new(self):
        sql = """
        {{ config(
            materialized='table'
        ) }}
        """
        new_desc = "New description"
        updated_sql = cli.DbtConfigManipulator.update_or_create(sql, new_desc)
        assert 'description = "New description"' in updated_sql
        assert "materialized='table'" in updated_sql

@pytest.mark.asyncio
class TestResolveDescription:
    @pytest.fixture
    def mock_db(self):
        db = MagicMock()
        db.get = AsyncMock()
        db.save = AsyncMock()
        return db

    async def test_resolve_keep_human_written(self, mock_db):
        # Scenario: Description exists and does not have AI tag
        current_desc = "Human written description"
        model_name = "my_model"
        col_name = "my_col"
        
        result = await cli.resolve_description(current_desc, model_name, col_name, mock_db, use_ai=True)
        
        assert result == current_desc
        mock_db.save.assert_called_with(model_name, col_name, current_desc)
        # Should NOT call AI
        # We can't easily verify ask_gemini wasn't called unless we mock it too, 
        # but checking logic flow implies it returns early.

    async def test_resolve_restore_from_db_if_human(self, mock_db):
        # Scenario: Description is missing, but DB has a human version
        current_desc = None
        model_name = "my_model"
        col_name = "my_col"
        cached_desc = "Cached human description"
        
        mock_db.get.return_value = cached_desc
        
        result = await cli.resolve_description(current_desc, model_name, col_name, mock_db, use_ai=True)
        
        assert result == cached_desc

    @patch('dbt_autodoc.cli.ask_gemini', new_callable=AsyncMock)
    async def test_resolve_generate_ai(self, mock_ask_gemini, mock_db):
        # Scenario: Description missing, nothing in cache (or cache is AI but we force regen logic? No, see logic)
        # Logic says: 
        # 1. Keep Human -> No
        # 2. Keep Existing AI -> No
        # 3. Restore from DB -> Returns None
        # 4. Ask AI -> Yes
        
        current_desc = None
        model_name = "my_model"
        col_name = "my_col"
        
        mock_db.get.return_value = None
        mock_ask_gemini.return_value = "AI generated description (ai_generated)"
        
        result = await cli.resolve_description(current_desc, model_name, col_name, mock_db, use_ai=True)
        
        assert result == "AI generated description (ai_generated)"
        mock_db.save.assert_called_with(model_name, col_name, result)

class TestGetDbtProjectInfo:
    @patch("builtins.open", new_callable=MagicMock)
    @patch("dbt_autodoc.cli.yaml")
    def test_get_dbt_project_info_success(self, mock_yaml, mock_open):
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file
        mock_yaml.load.return_value = {"name": "test_project", "profile": "test_profile"}
        
        info = cli.get_dbt_project_info()
        assert info["name"] == "test_project"
        assert info["profile"] == "test_profile"

    @patch("builtins.open", side_effect=FileNotFoundError)
    def test_get_dbt_project_info_missing_file(self, mock_open):
        info = cli.get_dbt_project_info()
        assert info["name"] == "unknown_project"
