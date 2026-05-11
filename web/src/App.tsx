import { Link, NavLink, Route, Routes } from "react-router-dom";
import { AnalyzePage } from "./pages/Analyze";
import { ComparePage } from "./pages/Compare";
import { CompareDetailPage } from "./pages/CompareDetail";
import { EvalDetailPage } from "./pages/EvalDetail";
import { EvalsPage } from "./pages/Evals";
import { GameReplayPage } from "./pages/GameReplay";
import { GamesPage } from "./pages/Games";
import { HomePage } from "./pages/Home";
import { PlayersPage } from "./pages/Players";
import { SpecsPage } from "./pages/Specs";

const navCls = ({ isActive }: { isActive: boolean }) =>
  `px-3 h-12 flex items-center text-sm border-b-2 -mb-px transition-colors ${
    isActive
      ? "border-neutral-900 text-neutral-900 font-medium"
      : "border-transparent text-neutral-500 hover:text-neutral-900"
  }`;

export function App() {
  return (
    <div className="min-h-full">
      <header className="border-b border-neutral-200 bg-white sticky top-0 z-10">
        <div className="max-w-6xl mx-auto px-6 flex items-center h-12 gap-8">
          <Link
            to="/"
            className="font-mono text-sm text-neutral-900 font-semibold tracking-tight hover:text-neutral-600 transition-colors"
          >
            main-street
          </Link>
          <nav className="flex items-center h-12">
            <NavLink to="/analyze" className={navCls}>Analyze</NavLink>
            <NavLink to="/compare" className={navCls}>Compare</NavLink>
            <NavLink to="/specs" className={navCls}>Specs</NavLink>
            <NavLink to="/players" className={navCls}>Players</NavLink>
            <NavLink to="/games" className={navCls}>Games</NavLink>
            <NavLink to="/evals" className={navCls}>Evals</NavLink>
          </nav>
        </div>
      </header>
      <main>
        <div className="max-w-6xl mx-auto px-6 py-10">
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/analyze" element={<AnalyzePage />} />
            <Route path="/compare" element={<ComparePage />} />
            <Route path="/compare/:id" element={<CompareDetailPage />} />
            <Route path="/specs" element={<SpecsPage />} />
            <Route path="/players" element={<PlayersPage />} />
            <Route path="/games" element={<GamesPage />} />
            <Route path="/games/:id" element={<GameReplayPage />} />
            <Route path="/evals" element={<EvalsPage />} />
            <Route path="/evals/:id" element={<EvalDetailPage />} />
            <Route path="*" element={<NotFound />} />
          </Routes>
        </div>
      </main>
    </div>
  );
}

function NotFound() {
  return (
    <div className="text-sm text-neutral-500">
      Not found.{" "}
      <Link to="/" className="text-neutral-900 underline underline-offset-2 hover:text-black hover:decoration-2">
        home
      </Link>
    </div>
  );
}
