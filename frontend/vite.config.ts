import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": process.env.COVERAGE_API_PROXY ?? "http://localhost:8000"
    }
  }
});
