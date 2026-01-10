/**
 * UI Context - Global UI state (activeTab, loading, errors)
 */

import React, { createContext, useContext, useReducer, ReactNode, Dispatch } from 'react';

export type TabId =
  | 'testing'
  | 'orchestration'
  | 'workflow'
  | 'workflow-playground'
  | 'llm'
  | 'rag'
  | 'tools'
  | 'state'
  | 'monitoring'
  | 'safety'
  | 'chat'
  | 'health'
  | 'history'
  | 'vendor-fraud'
  | 'agent-playground';

interface UIState {
  activeTab: TabId;
  loading: string | null;
  error: string | null;
  sidebarCollapsed: boolean;
}

type UIAction =
  | { type: 'SET_ACTIVE_TAB'; payload: TabId }
  | { type: 'SET_LOADING'; payload: string | null }
  | { type: 'SET_ERROR'; payload: string | null }
  | { type: 'CLEAR_ERROR' }
  | { type: 'TOGGLE_SIDEBAR' }
  | { type: 'SET_SIDEBAR_COLLAPSED'; payload: boolean };

interface UIContextValue {
  state: UIState;
  dispatch: Dispatch<UIAction>;
}

const UIContext = createContext<UIContextValue | null>(null);

const initialState: UIState = {
  activeTab: 'orchestration',
  loading: null,
  error: null,
  sidebarCollapsed: false
};

function uiReducer(state: UIState, action: UIAction): UIState {
  switch (action.type) {
    case 'SET_ACTIVE_TAB':
      return { ...state, activeTab: action.payload };
    case 'SET_LOADING':
      return { ...state, loading: action.payload };
    case 'SET_ERROR':
      return { ...state, error: action.payload };
    case 'CLEAR_ERROR':
      return { ...state, error: null };
    case 'TOGGLE_SIDEBAR':
      return { ...state, sidebarCollapsed: !state.sidebarCollapsed };
    case 'SET_SIDEBAR_COLLAPSED':
      return { ...state, sidebarCollapsed: action.payload };
    default:
      return state;
  }
}

export function UIProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(uiReducer, initialState);

  return (
    <UIContext.Provider value={{ state, dispatch }}>
      {children}
    </UIContext.Provider>
  );
}

export function useUI() {
  const context = useContext(UIContext);
  if (!context) {
    throw new Error('useUI must be used within UIProvider');
  }
  return context;
}
