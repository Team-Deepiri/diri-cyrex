/**
 * AppProviders - Wrapper component for all context providers
 */

import React, { ReactNode } from 'react';
import { UIProvider } from './UIContext';

interface AppProvidersProps {
  children: ReactNode;
}

export function AppProviders({ children }: AppProvidersProps) {
  return (
    <UIProvider>
      {children}
    </UIProvider>
  );
}
