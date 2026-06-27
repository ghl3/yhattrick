import { Link, Outlet } from "react-router-dom";

export default function App() {
  return (
    <>
      <header className="app-header">
        <h1>
          <Link to="/">Hockey Data Explorer</Link>
        </h1>
        <div className="sub">Per-game stint &amp; event inspection — on-ice reconstruction vs. borrowed xG</div>
      </header>
      <div className="container">
        <Outlet />
      </div>
    </>
  );
}
