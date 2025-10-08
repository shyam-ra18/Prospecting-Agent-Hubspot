/** @type {import('tailwindcss').Config} */
export default { // Note the 'export default' for Vite
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'agent-blue': '#1e3a8a', // Dark Blue
        'agent-green': '#10b981', // Emerald Green
        'agent-yellow': '#f59e0b', // Amber Yellow
      },
    },
  },
  plugins: [],
}