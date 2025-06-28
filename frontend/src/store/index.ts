import { configureStore } from '@reduxjs/toolkit';

// Simple store for basic functionality
export const store = configureStore({
  reducer: {
    app: (state = { isLoading: false, theme: 'dark' }) => state
  },
  devTools: import.meta.env.MODE !== 'production'
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;