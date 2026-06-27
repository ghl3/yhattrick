import { Link, NavLink, Outlet } from "react-router-dom";

export default function App() {
  return (
    <>
      <header className="navbar">
        <div className="nav-inner">
          <Link to="/" className="brand">
            <svg width="24" height="24" viewBox="0 0 24 24" aria-hidden="true">
              <rect x="0" y="0" width="24" height="24" rx="6" fill="#2f6cb0" />
              <ellipse cx="12" cy="12" rx="6.5" ry="3.6" fill="#fff" />
              <ellipse cx="12" cy="12" rx="6.5" ry="3.6" fill="none" stroke="#cfe0f0" strokeWidth="0.8" />
            </svg>
            <span className="brand-name">Hockey Data Explorer</span>
          </Link>
          <nav className="nav-links">
            <NavLink to="/" end>
              Games
            </NavLink>
            <NavLink to="/players">Players</NavLink>
          </nav>
        </div>
      </header>
      <div className="container">
        <Outlet />
      </div>
    </>
  );
}
