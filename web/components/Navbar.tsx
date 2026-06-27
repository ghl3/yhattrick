"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";

export default function Navbar() {
  const path = usePathname();
  const isActive = (href: string, exact = false) =>
    exact ? path === href : path === href || path.startsWith(href + "/");

  return (
    <header className="navbar">
      <div className="nav-inner">
        <Link href="/" className="brand">
          <span className="brand-name"><span className="brand-y">ŷ</span>Trick</span>
        </Link>
        <nav className="nav-links">
          <Link className={isActive("/", true) ? "active" : ""} href="/">
            Games
          </Link>
          <Link className={isActive("/players") ? "active" : ""} href="/players">
            Players
          </Link>
          <Link className={isActive("/about") ? "active" : ""} href="/about">
            About
          </Link>
        </nav>
      </div>
    </header>
  );
}
