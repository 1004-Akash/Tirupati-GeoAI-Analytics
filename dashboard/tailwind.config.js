/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        nature: {
          50: '#f2f9f2',
          100: '#e1f1e1',
          500: '#228B22',
          900: '#1a4d1a',
        },
        urban: {
          500: '#FF4500',
        },
        water: {
          500: '#1E90FF',
        },
        agri: {
          500: '#DAA520',
        },
        barren: {
          500: '#DEB887',
        }
      }
    },
  },
  plugins: [],
}
