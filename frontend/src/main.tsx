import { createRoot } from 'react-dom/client'
import App from './App.tsx'
import "@fontsource/plus-jakarta-sans/400.css";
import "@fontsource/plus-jakarta-sans/700.css";
import "@fontsource/plus-jakarta-sans/800.css";
import './index.css'

createRoot(document.getElementById("root")!).render(<App />);
