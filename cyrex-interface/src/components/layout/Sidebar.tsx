/**
 * Sidebar component - Collapsible navigation sidebar
 */

import React from 'react';
import {
  FaFlask,
  FaRandom,
  FaProjectDiagram,
  FaServer,
  FaDatabase,
  FaTools,
  FaCog,
  FaChartLine,
  FaShieldAlt,
  FaComments,
  FaHistory,
  FaHeartbeat,
  FaChevronLeft,
  FaChevronRight,
  FaSearchDollar,
  FaRobot,
  FaSitemap
} from 'react-icons/fa';
import { useUI, type TabId } from '../../context/UIContext';
import styles from './Sidebar.module.css';

interface NavigationItem {
  id: TabId;
  label: string;
  icon: React.ReactNode;
}

const navigationItems: NavigationItem[] = [
  { id: 'agent-playground', label: 'Agent Playground', icon: <FaRobot /> },
  { id: 'testing', label: 'Infrastructure Suite', icon: <FaFlask /> },
  { id: 'orchestration', label: 'Orchestration', icon: <FaRandom /> },
  { id: 'workflow', label: 'Workflows', icon: <FaProjectDiagram /> },
  { id: 'workflow-playground', label: 'LangGraph Workflow', icon: <FaSitemap /> },
  { id: 'llm', label: 'Local LLM', icon: <FaServer /> },
  { id: 'rag', label: 'RAG / Vector Store', icon: <FaDatabase /> },
  { id: 'tools', label: 'Tools', icon: <FaTools /> },
  { id: 'state', label: 'State Management', icon: <FaCog /> },
  { id: 'monitoring', label: 'Monitoring', icon: <FaChartLine /> },
  { id: 'safety', label: 'Safety / Guardrails', icon: <FaShieldAlt /> },
  { id: 'chat', label: 'Chat', icon: <FaComments /> },
  { id: 'vendor-fraud', label: 'Vendor Fraud Detection', icon: <FaSearchDollar /> },
  { id: 'health', label: 'Health', icon: <FaHeartbeat /> },
  { id: 'history', label: 'History', icon: <FaHistory /> }
];

export function Sidebar() {
  const { state, dispatch } = useUI();

  const handleItemClick = (tabId: TabId) => {
    dispatch({ type: 'SET_ACTIVE_TAB', payload: tabId });
  };

  const toggleSidebar = () => {
    dispatch({ type: 'TOGGLE_SIDEBAR' });
  };

  return (
    <aside className={`${styles.sidebar} ${state.sidebarCollapsed ? styles.collapsed : ''}`}>
      <div className={styles.header}>
        <div className={styles.logo}>
          <img src="/logo.png" alt="Cyrex Logo" className={styles.logoImage} />
          <span className={styles.logoText}>
            <span className={styles.logoBrand}>CYREX</span>
            <span className={styles.logoSubtitle}> Testing UI</span>
          </span>
        </div>
      </div>

      <nav className={styles.navigation}>
        {navigationItems.map((item) => (
          <button
            key={item.id}
            onClick={() => handleItemClick(item.id)}
            className={`${styles.navItem} ${state.activeTab === item.id ? styles.active : ''}`}
            title={state.sidebarCollapsed ? item.label : undefined}
          >
            <span className={styles.navIcon}>{item.icon}</span>
            {!state.sidebarCollapsed && <span className={styles.navLabel}>{item.label}</span>}
          </button>
        ))}
      </nav>

      <div className={styles.footer}>
        <button
          className={styles.toggleButton}
          onClick={toggleSidebar}
          aria-label={state.sidebarCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
        >
          {state.sidebarCollapsed ? <FaChevronRight /> : <FaChevronLeft />}
        </button>
      </div>
    </aside>
  );
}
