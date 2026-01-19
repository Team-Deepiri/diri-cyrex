"""
Spreadsheet Tools for Agents
Tools for interacting with the live spreadsheet in the agent playground
"""
from typing import Dict, Any, Optional
import json
from ...logging_config import get_logger
from ...database.postgres import get_postgres_manager

logger = get_logger("cyrex.agent.tools.spreadsheet")

# Global spreadsheet state (in production, this would be per-instance)
_spreadsheet_states: Dict[str, Dict[str, Any]] = {}


async def _get_spreadsheet_data(instance_id: str, user_id: str = "admin") -> Dict[str, Any]:
    """Get spreadsheet data from PostgreSQL"""
    try:
        postgres = await get_postgres_manager()
        spreadsheet_id = f"{user_id}_{instance_id or 'default'}"
        
        row = await postgres.fetchrow("""
            SELECT data, columns, row_count
            FROM cyrex.spreadsheet_data
            WHERE spreadsheet_id = $1
        """, spreadsheet_id)
        
        if row:
            data = json.loads(row['data']) if isinstance(row['data'], str) else row['data']
            columns = json.loads(row['columns']) if isinstance(row['columns'], str) else row['columns']
            return {"data": data, "columns": columns, "row_count": row['row_count']}
        return {"data": {}, "columns": [], "row_count": 20}
    except Exception as e:
        logger.warning(f"Failed to load spreadsheet from DB: {e}, using in-memory state")
        return {"data": _spreadsheet_states.get(instance_id, {}), "columns": [], "row_count": 20}


async def _save_spreadsheet_data(instance_id: str, data: Dict[str, Any], user_id: str = "admin"):
    """Save spreadsheet data to PostgreSQL"""
    try:
        postgres = await get_postgres_manager()
        spreadsheet_id = f"{user_id}_{instance_id or 'default'}"
        
        # Get current columns and row_count
        row = await postgres.fetchrow("""
            SELECT columns, row_count
            FROM cyrex.spreadsheet_data
            WHERE spreadsheet_id = $1
        """, spreadsheet_id)
        
        columns = json.loads(row['columns']) if row and isinstance(row['columns'], str) else (row['columns'] if row else ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'])
        row_count = row['row_count'] if row else 20
        
        await postgres.execute("""
            INSERT INTO cyrex.spreadsheet_data 
            (spreadsheet_id, user_id, instance_id, columns, row_count, data, updated_at)
            VALUES ($1, $2, $3, $4, $5, $6, NOW())
            ON CONFLICT (spreadsheet_id) DO UPDATE SET
                data = EXCLUDED.data,
                updated_at = NOW()
        """,
            spreadsheet_id,
            user_id,
            instance_id,
            json.dumps(columns),
            row_count,
            json.dumps(data),
        )
    except Exception as e:
        logger.warning(f"Failed to save spreadsheet to DB: {e}, using in-memory state")
        if instance_id not in _spreadsheet_states:
            _spreadsheet_states[instance_id] = {}
        _spreadsheet_states[instance_id] = data


async def register_spreadsheet_tools(agent, instance_id: Optional[str] = None):
    """Register spreadsheet tools with an agent"""
    
    async def set_cell(instance_id: str, cell_id: str, value: str) -> Dict[str, Any]:
        """
        Set a cell value in the spreadsheet
        
        Args:
            instance_id: The agent instance ID
            cell_id: Cell identifier (e.g., "A1", "B2", "J7")
            value: Value to set (can be a number or formula starting with =)
        
        Returns:
            Success status and cell info
        """
        try:
            # Get current spreadsheet data
            spreadsheet = await _get_spreadsheet_data(instance_id)
            data = spreadsheet["data"]
            
            # Update cell
            if cell_id not in data:
                data[cell_id] = {"id": cell_id, "value": ""}
            
            cell = data[cell_id]
            if value.startswith("="):
                cell["formula"] = value[1:]
                cell["value"] = ""
            else:
                cell["value"] = value
                cell["formula"] = None
            
            # Save to database
            await _save_spreadsheet_data(instance_id, data)
            
            logger.info(f"Set cell {cell_id} to {value} for instance {instance_id}")
            return {
                "success": True,
                "cell_id": cell_id,
                "value": value,
                "message": f"Set cell {cell_id} to {value}",
            }
        except Exception as e:
            logger.error(f"Failed to set cell: {e}")
            return {
                "success": False,
                "error": str(e),
            }
    
    async def get_cell(instance_id: str, cell_id: str) -> Dict[str, Any]:
        """
        Get a cell value from the spreadsheet
        
        Args:
            instance_id: The agent instance ID
            cell_id: Cell identifier (e.g., "A1", "B2", "J7")
        
        Returns:
            Cell value and metadata
        """
        try:
            # Get spreadsheet data from database
            spreadsheet = await _get_spreadsheet_data(instance_id)
            data = spreadsheet["data"]
            
            cell = data.get(cell_id, {})
            
            return {
                "success": True,
                "cell_id": cell_id,
                "value": cell.get("value", ""),
                "formula": cell.get("formula"),
                "computedValue": cell.get("computedValue"),
            }
        except Exception as e:
            logger.error(f"Failed to get cell: {e}")
            return {
                "success": False,
                "error": str(e),
            }
    
    def sum_range(instance_id: str, start_cell: str, end_cell: str, target_cell: Optional[str] = None) -> Dict[str, Any]:
        """
        Calculate sum of a range of cells
        
        Args:
            instance_id: The agent instance ID
            start_cell: Starting cell (e.g., "A1")
            end_cell: Ending cell (e.g., "A10")
            target_cell: Optional cell to store the result
        
        Returns:
            Sum result and optionally sets target cell
        """
        try:
            if instance_id not in _spreadsheet_states:
                return {
                    "success": False,
                    "error": "Spreadsheet not initialized for this instance",
                }
            
            state = _spreadsheet_states[instance_id]
            
            # Parse range
            start_col = start_cell[0]
            start_row = int(start_cell[1:])
            end_col = end_cell[0]
            end_row = int(end_cell[1:])
            
            # Calculate sum
            total = 0.0
            count = 0
            
            for col in range(ord(start_col), ord(end_col) + 1):
                for row in range(start_row, end_row + 1):
                    cell_id = f"{chr(col)}{row}"
                    cell = state.get(cell_id, {})
                    value = cell.get("value", "")
                    
                    try:
                        num = float(value) if value else 0.0
                        total += num
                        count += 1
                    except ValueError:
                        pass
            
            # Store result if target cell provided
            if target_cell:
                state[target_cell] = {
                    "value": str(total),
                    "formula": f"SUM({start_cell}:{end_cell})",
                }
            
            logger.info(f"Sum of {start_cell}:{end_cell} = {total} for instance {instance_id}")
            return {
                "success": True,
                "sum": total,
                "range": f"{start_cell}:{end_cell}",
                "count": count,
                "target_cell": target_cell,
            }
        except Exception as e:
            logger.error(f"Failed to sum range: {e}")
            return {
                "success": False,
                "error": str(e),
            }
    
    def avg_range(instance_id: str, start_cell: str, end_cell: str, target_cell: Optional[str] = None) -> Dict[str, Any]:
        """
        Calculate average of a range of cells
        
        Args:
            instance_id: The agent instance ID
            start_cell: Starting cell (e.g., "A1")
            end_cell: Ending cell (e.g., "A10")
            target_cell: Optional cell to store the result
        
        Returns:
            Average result and optionally sets target cell
        """
        try:
            if instance_id not in _spreadsheet_states:
                return {
                    "success": False,
                    "error": "Spreadsheet not initialized for this instance",
                }
            
            state = _spreadsheet_states[instance_id]
            
            # Parse range
            start_col = start_cell[0]
            start_row = int(start_cell[1:])
            end_col = end_cell[0]
            end_row = int(end_cell[1:])
            
            # Calculate average
            total = 0.0
            count = 0
            
            for col in range(ord(start_col), ord(end_col) + 1):
                for row in range(start_row, end_row + 1):
                    cell_id = f"{chr(col)}{row}"
                    cell = state.get(cell_id, {})
                    value = cell.get("value", "")
                    
                    try:
                        num = float(value) if value else 0.0
                        total += num
                        count += 1
                    except ValueError:
                        pass
            
            avg = total / count if count > 0 else 0.0
            
            # Store result if target cell provided
            if target_cell:
                state[target_cell] = {
                    "value": str(avg),
                    "formula": f"AVG({start_cell}:{end_cell})",
                }
            
            logger.info(f"Average of {start_cell}:{end_cell} = {avg} for instance {instance_id}")
            return {
                "success": True,
                "average": avg,
                "range": f"{start_cell}:{end_cell}",
                "count": count,
                "target_cell": target_cell,
            }
        except Exception as e:
            logger.error(f"Failed to calculate average: {e}")
            return {
                "success": False,
                "error": str(e),
            }
    
    def add_row(instance_id: str) -> Dict[str, Any]:
        """
        Add a new row to the spreadsheet
        
        Args:
            instance_id: The agent instance ID
        
        Returns:
            Success status
        """
        try:
            if instance_id not in _spreadsheet_states:
                _spreadsheet_states[instance_id] = {}
            
            # This is a notification - actual row addition handled by frontend
            logger.info(f"Row addition requested for instance {instance_id}")
            return {
                "success": True,
                "message": "Row addition requested",
                "action": "add_row",
            }
        except Exception as e:
            logger.error(f"Failed to add row: {e}")
            return {
                "success": False,
                "error": str(e),
            }
    
    def add_column(instance_id: str) -> Dict[str, Any]:
        """
        Add a new column to the spreadsheet
        
        Args:
            instance_id: The agent instance ID
        
        Returns:
            Success status
        """
        try:
            if instance_id not in _spreadsheet_states:
                _spreadsheet_states[instance_id] = {}
            
            # This is a notification - actual column addition handled by frontend
            logger.info(f"Column addition requested for instance {instance_id}")
            return {
                "success": True,
                "message": "Column addition requested",
                "action": "add_column",
            }
        except Exception as e:
            logger.error(f"Failed to add column: {e}")
            return {
                "success": False,
                "error": str(e),
            }
    
    # Create wrapper functions that capture instance_id
    # These are async-compatible wrappers
    async def make_set_cell_async(cell_id: str, value: str):
        return await set_cell(instance_id or "default", cell_id, value)
    
    async def make_get_cell_async(cell_id: str):
        return await get_cell(instance_id or "default", cell_id)
    
    # Sync wrappers for backward compatibility (fallback to in-memory if async fails)
    def make_set_cell(cell_id: str, value: str):
        # Try async first, fallback to sync in-memory
        try:
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Can't use run() in running loop, use in-memory fallback
                    raise RuntimeError("Event loop running")
            except RuntimeError:
                pass
            
            # Try to run async
            return asyncio.run(set_cell(instance_id or "default", cell_id, value))
        except Exception:
            # Fallback to in-memory
            if instance_id not in _spreadsheet_states:
                _spreadsheet_states[instance_id] = {}
            _spreadsheet_states[instance_id][cell_id] = {"value": value, "formula": value[1:] if value.startswith("=") else None}
            return {"success": True, "cell_id": cell_id, "value": value, "message": f"Set cell {cell_id} to {value}"}
    
    def make_get_cell(cell_id: str):
        # Try async first, fallback to sync in-memory
        try:
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    raise RuntimeError("Event loop running")
            except RuntimeError:
                pass
            
            return asyncio.run(get_cell(instance_id or "default", cell_id))
        except Exception:
            # Fallback to in-memory
            if instance_id not in _spreadsheet_states:
                return {"success": False, "error": "Spreadsheet not initialized"}
            cell = _spreadsheet_states[instance_id].get(cell_id, {})
            return {"success": True, "cell_id": cell_id, "value": cell.get("value", ""), "formula": cell.get("formula")}
    
    def make_sum_range(start_cell: str, end_cell: str, target_cell: Optional[str] = None):
        return sum_range(instance_id or "default", start_cell, end_cell, target_cell)
    
    def make_avg_range(start_cell: str, end_cell: str, target_cell: Optional[str] = None):
        return avg_range(instance_id or "default", start_cell, end_cell, target_cell)
    
    def make_add_row():
        return add_row(instance_id or "default")
    
    def make_add_column():
        return add_column(instance_id or "default")
    
    # Register tools with agent
    agent.register_tool(
        "spreadsheet_set_cell",
        make_set_cell,
        "Set a cell value in the spreadsheet. Use cell_id like 'A1', 'B2', etc. Value can be a number or formula starting with ="
    )
    
    agent.register_tool(
        "spreadsheet_get_cell",
        make_get_cell,
        "Get a cell value from the spreadsheet. Use cell_id like 'A1', 'B2', etc."
    )
    
    agent.register_tool(
        "spreadsheet_sum_range",
        make_sum_range,
        "Calculate sum of a range of cells. Provide start_cell and end_cell like 'A1' and 'A10'. Optionally provide target_cell to store result."
    )
    
    agent.register_tool(
        "spreadsheet_avg_range",
        make_avg_range,
        "Calculate average of a range of cells. Provide start_cell and end_cell like 'A1' and 'A10'. Optionally provide target_cell to store result."
    )
    
    agent.register_tool(
        "spreadsheet_add_row",
        make_add_row,
        "Add a new row to the spreadsheet"
    )
    
    agent.register_tool(
        "spreadsheet_add_column",
        make_add_column,
        "Add a new column to the spreadsheet"
    )


def get_spreadsheet_state(instance_id: str) -> Dict[str, Any]:
    """Get spreadsheet state for an instance"""
    return _spreadsheet_states.get(instance_id, {})


def set_spreadsheet_state(instance_id: str, state: Dict[str, Any]):
    """Set spreadsheet state for an instance"""
    _spreadsheet_states[instance_id] = state

