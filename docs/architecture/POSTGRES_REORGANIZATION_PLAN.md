# PostgreSQL Schema Reorganization Plan

## 🎯 Goal
Separate AI/Agent database tables from Backend/User database tables using schemas.

---

## 📊 Current State (MESSY!)

### **Backend/User Database** (in `postgres-init.sql`):
**Schema: `public`**
- users, roles, sessions, projects, tasks, quests, seasons, rewards, etc.

**Schema: `analytics`**
- momentum, streaks, boosts, achievements, level_progress

**Schema: `audit`**
- activity_logs, task_completions, user_activity_summary

### **AI/Agent Database** (scattered in cyrex codebase, all in `public` schema!):
- `agent_playground_messages` (agent_tables.py)
- `workflows` (agent_tables.py)
- `task_executions` (agent_tables.py)
- `events` (agent_tables.py)
- `cyrex_sessions` (session_manager.py)
- `guardrail_rules`, `guardrail_violations` (enhanced_guardrails.py)
- `agents`, `agent_states` (agent_initializer.py)
- `cyrex_vendors`, `cyrex_invoices`, `cyrex_pricing_benchmarks` (vendor_intelligence_service.py)
- `events` (event_handler.py - duplicate?)

**Problem:** All AI tables are in `public` schema mixed with user data! 😱

---

## ✅ Proposed Organization

### **New Schema: `cyrex`** (AI/Agent System)

Move all AI/agent tables to `cyrex` schema:

```
cyrex schema:
├── agent_playground_messages  (conversations)
├── workflows                  (workflow state)
├── task_executions            (task execution state)
├── events                     (event audit log)
├── cyrex_sessions             (AI sessions)
├── guardrail_rules            (guardrail configs)
├── guardrail_violations       (violation logs)
├── agents                     (agent configs)
├── agent_states               (agent state)
├── cyrex_vendors              (vendor intelligence)
├── cyrex_invoices             (invoice data)
└── cyrex_pricing_benchmarks   (pricing data)
```

### **Keep Existing Schemas:**

**`public` schema:** Backend/User data only
- users, roles, sessions, projects, tasks, quests, seasons, etc.

**`analytics` schema:** Gamification data
- momentum, streaks, boosts, achievements

**`audit` schema:** Audit logs
- activity_logs, task_completions, user_activity_summary

---

## 🔄 Migration Plan

### Step 1: Create `cyrex` Schema
```sql
CREATE SCHEMA IF NOT EXISTS cyrex;
```

### Step 2: Move Tables to `cyrex` Schema

**Option A: Rename (if tables exist)**
```sql
ALTER TABLE agent_playground_messages SET SCHEMA cyrex;
ALTER TABLE workflows SET SCHEMA cyrex;
ALTER TABLE task_executions SET SCHEMA cyrex;
ALTER TABLE events SET SCHEMA cyrex;
ALTER TABLE cyrex_sessions SET SCHEMA cyrex;
ALTER TABLE guardrail_rules SET SCHEMA cyrex;
ALTER TABLE guardrail_violations SET SCHEMA cyrex;
ALTER TABLE agents SET SCHEMA cyrex;
ALTER TABLE agent_states SET SCHEMA cyrex;
ALTER TABLE cyrex_vendors SET SCHEMA cyrex;
ALTER TABLE cyrex_invoices SET SCHEMA cyrex;
ALTER TABLE cyrex_pricing_benchmarks SET SCHEMA cyrex;
```

**Option B: Create in `cyrex` schema (if tables don't exist yet)**
- Update all `CREATE TABLE` statements to use `cyrex.` prefix

### Step 3: Update Code References
- Update all queries to use `cyrex.` schema prefix
- Update table creation code in Python files

---

## 📋 Files to Update

1. **`app/database/agent_tables.py`**
   - Update `CREATE TABLE` to use `cyrex.` schema

2. **`app/core/session_manager.py`**
   - Update `cyrex_sessions` table to `cyrex.cyrex_sessions`

3. **`app/core/enhanced_guardrails.py`**
   - Update `guardrail_rules` and `guardrail_violations` to `cyrex.` schema

4. **`app/core/agent_initializer.py`**
   - Update `agents` and `agent_states` to `cyrex.` schema

5. **`app/services/vendor_intelligence_service.py`**
   - Update `cyrex_vendors`, `cyrex_invoices`, `cyrex_pricing_benchmarks` to `cyrex.` schema

6. **`app/core/event_handler.py`**
   - Check if `events` table conflicts with `agent_tables.py`
   - Move to `cyrex.` schema

7. **All query references**
   - Update all `SELECT`, `INSERT`, `UPDATE` queries to use `cyrex.` prefix

---

## 🎯 Final Schema Structure

```
PostgreSQL Database
├── public (Backend/User)
│   ├── users
│   ├── roles
│   ├── sessions
│   ├── projects
│   ├── tasks
│   ├── quests
│   └── ...
├── analytics (Gamification)
│   ├── momentum
│   ├── streaks
│   ├── boosts
│   └── ...
├── audit (Audit Logs)
│   ├── activity_logs
│   ├── task_completions
│   └── ...
└── cyrex (AI/Agent System) ← NEW!
    ├── agent_playground_messages
    ├── workflows
    ├── task_executions
    ├── events
    ├── cyrex_sessions
    ├── guardrail_rules
    ├── guardrail_violations
    ├── agents
    ├── agent_states
    ├── cyrex_vendors
    ├── cyrex_invoices
    └── cyrex_pricing_benchmarks
```

---

## ✅ Benefits

1. **Clear Separation**: AI/Agent data separate from user data
2. **Easy to Query**: `SELECT * FROM cyrex.workflows` vs `SELECT * FROM public.tasks`
3. **Permissions**: Can set different permissions per schema
4. **Backup**: Can backup schemas separately
5. **Organization**: Much cleaner and easier to understand

Want me to implement this reorganization?

