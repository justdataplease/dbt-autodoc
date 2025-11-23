import pytest
from unittest.mock import MagicMock, patch, AsyncMock, call
from dbt_autodoc import cli

class TestDatabaseAdapter:
    @patch("duckdb.connect")
    def test_init_table_duckdb(self, mock_connect):
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        
        db = cli.DatabaseAdapter(project_info={"name": "test", "profile": "test"})
        db.type = 'duckdb' # Ensure type is duckdb
        db.connect()
        
        # We mock migrate_schema to avoid complex interactions or let it run?
        # Let's mock it to focus on init_table logic
        with patch.object(db, 'migrate_schema', new_callable=MagicMock) as mock_migrate:
            mock_migrate.return_value = False # No reset needed
            db.init_table()
            
            # Check if create table statements were executed
            assert mock_conn.execute.call_count >= 2
            calls = mock_conn.execute.call_args_list
            create_cache = False
            create_log = False
            for c in calls:
                sql = c[0][0]
                if "CREATE TABLE IF NOT EXISTS doc_cache" in sql: create_cache = True
                if "CREATE TABLE IF NOT EXISTS doc_cache_log" in sql: create_log = True
            
            assert create_cache
            assert create_log
            mock_migrate.assert_called_once()

    @patch("duckdb.connect")
    def test_save_duckdb(self, mock_connect):
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        
        db = cli.DatabaseAdapter(project_info={"name": "p1", "profile": "prof1"})
        db.type = 'duckdb'
        db.connect()
        
        # Mock get to return None (new entry)
        with patch.object(db, 'get', new_callable=MagicMock) as mock_get:
            mock_get.return_value = None
            
            # Mock log_change to do nothing
            with patch.object(db, 'log_change', new_callable=MagicMock) as mock_log:
                db.save("my_model", "my_col", "desc")
                
                # Check execute called via cursor (since new code uses cursor())
                # We mocked mock_connect.return_value = mock_conn
                # Code calls self.conn.cursor().execute(...)
                # So we should check if mock_conn.cursor.return_value.execute was called
                
                assert mock_conn.cursor.return_value.execute.called
                args, _ = mock_conn.cursor.return_value.execute.call_args
                assert "INSERT OR REPLACE INTO doc_cache" in args[0]
                # Check params
                params = args[1]
                # (dbt_project_name, dbt_profile_name, model_name, column_name, description, user_name, is_human)
                assert params[0] == "p1"
                assert params[1] == "prof1"
                assert params[2] == "my_model"
                assert params[3] == "my_col"
                assert params[4] == "desc"

    @patch("duckdb.connect")
    def test_get_duckdb(self, mock_connect):
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        # Mock fetchone result
        # Code: self.conn.cursor().execute(q, params).fetchone()
        mock_conn.cursor.return_value.execute.return_value.fetchone.return_value = ["existing_desc"]
        
        db = cli.DatabaseAdapter(project_info={"name": "p1", "profile": "prof1"})
        db.type = 'duckdb'
        db.connect()
        
        val = db.get("my_model", "my_col")
        assert val == "existing_desc"
        
        # Verify query
        args, _ = mock_conn.cursor.return_value.execute.call_args
        assert "SELECT description FROM doc_cache" in args[0]
        assert args[1] == ("p1", "prof1", "my_model", "my_col")

    @patch("duckdb.connect")
    def test_cleanup_db_duckdb(self, mock_connect):
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        
        db = cli.DatabaseAdapter()
        db.type = 'duckdb'
        db.connect()
        
        # Mock input to return 'yes'
        with patch("builtins.input", return_value="yes"):
            cli.action_cleanup_db(db)
            
            # Check Drop statements
            calls = mock_conn.execute.call_args_list
            drops = [c[0][0] for c in calls if "DROP TABLE" in c[0][0]]
            assert len(drops) >= 2
            assert "DROP TABLE IF EXISTS doc_cache" in drops
            assert "DROP TABLE IF EXISTS doc_cache_log" in drops
