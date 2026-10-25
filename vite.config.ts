import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [
    react(),
  ],
  optimizeDeps: {
    include: [
      "three",
      "three/addons/geometries/RoundedBoxGeometry.js",
      "react",
      "react-dom",
      "react-dom/client",
      "react/jsx-runtime",
      "react/jsx-dev-runtime",
      "canvas-confetti",
      "lucide-react",
      "zustand",
      "@supabase/supabase-js",
      "animejs",
      "suncalc",
    ],
  },
  server: {
    port: 3000,
    host: "0.0.0.0",
    hmr: false,
  },
});
