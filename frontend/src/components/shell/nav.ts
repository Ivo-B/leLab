export interface NavItem {
  label: string;
  to: string;
  isActive: (pathname: string) => boolean;
}

function startsWith(pathname: string, prefix: string): boolean {
  return pathname === prefix || pathname.startsWith(`${prefix}/`);
}

export const NAV: NavItem[] = [
  {
    label: "Setup",
    to: "/setup",
    isActive: (p) => p === "/setup" || startsWith(p, "/calibration"),
  },
  {
    label: "Record",
    to: "/",
    isActive: (p) =>
      p === "/" ||
      ["/recording", "/teleoperation", "/upload", "/edit-dataset"].some((prefix) =>
        startsWith(p, prefix),
      ),
  },
  { label: "Train", to: "/training", isActive: (p) => startsWith(p, "/training") },
  {
    label: "Evaluate",
    to: "/evaluate",
    isActive: (p) => p === "/evaluate" || startsWith(p, "/inference"),
  },
  { label: "Orders", to: "/orders", isActive: (p) => p === "/orders" },
  { label: "Improve", to: "/improve", isActive: (p) => p === "/improve" },
];
