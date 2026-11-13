import { Link, Outlet, useLocation } from "react-router-dom";
import { BrandMark } from "./BrandMark";
import { NAV } from "./nav";
import { StatusBar } from "./StatusBar";

export function AppShell() {
  const { pathname } = useLocation();
  return (
    <div className="flex h-screen bg-background text-foreground">
      <aside className="flex w-56 shrink-0 flex-col gap-6 border-r border-border p-4">
        <BrandMark />
        <nav className="flex flex-col gap-1">
          {NAV.map((item) => {
            const active = item.isActive(pathname);
            return (
              <Link
                key={item.label}
                to={item.to}
                aria-current={active ? "page" : undefined}
                className={`rounded-md px-3 py-2 text-sm font-bold ${
                  active ? "bg-primary text-primary-foreground" : "hover:bg-accent"
                }`}
              >
                {item.label}
              </Link>
            );
          })}
        </nav>
      </aside>
      <div className="flex min-w-0 flex-1 flex-col">
        <StatusBar />
        <main className="min-h-0 flex-1 overflow-auto">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
