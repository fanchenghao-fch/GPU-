import Calculator from './components/Calculator';

export default function App() {
  return (
    <div className="min-h-screen flex flex-col">
      <div className="flex-1 flex items-start justify-center px-4 py-8 sm:py-16">
        <Calculator />
      </div>
      <footer className="pb-6 text-center text-xs text-slate-500">
        <a
          href="https://beian.miit.gov.cn/"
          target="_blank"
          rel="noopener noreferrer"
          className="hover:text-slate-700 transition-colors"
        >
          沪ICP备2026042468号-1
        </a>
      </footer>
    </div>
  );
}
