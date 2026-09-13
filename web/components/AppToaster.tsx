"use client";

import { Toaster } from "sonner";

export function AppToaster() {
  return (
    <Toaster
      theme="light"
      richColors
      closeButton
      position="top-center"
      toastOptions={{
        duration: 8000,
      }}
    />
  );
}
