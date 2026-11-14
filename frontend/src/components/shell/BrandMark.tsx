import { useEffect, useState } from "react";
import { cn } from "@/lib/utils";

const DARK_MARK = "/brand/cw-dark.png";
const LIGHT_MARK = "/brand/cw-light.png";

interface BrandMarkProps {
  iconOnly?: boolean;
  className?: string;
}

function useDarkSurface() {
  const [dark, setDark] = useState(true);

  useEffect(() => {
    const root = document.documentElement;
    const sync = () => setDark(!root.classList.contains("light"));
    sync();
    const observer = new MutationObserver(sync);
    observer.observe(root, { attributes: true, attributeFilter: ["class"] });
    return () => observer.disconnect();
  }, []);

  return dark;
}

export function BrandMark({ iconOnly = false, className }: BrandMarkProps) {
  const dark = useDarkSurface();
  const src = dark ? DARK_MARK : LIGHT_MARK;
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    setFailed(false);
  }, [src]);

  return (
    <div className={cn("flex items-center gap-2", className)}>
      {failed ? (
        <span
          data-testid="brand-fallback"
          aria-hidden
          className="grid h-8 w-8 place-items-center rounded-md bg-primary font-extrabold text-primary-foreground"
        >
          G
        </span>
      ) : (
        <img
          src={src}
          alt="CoreWeave"
          className="h-8 w-auto"
          onError={() => setFailed(true)}
        />
      )}
      {!iconOnly && (
        <span className="flex flex-col leading-none">
          <span className="font-extrabold text-xl">Garage</span>
          <span className="mt-1 text-xs text-muted-foreground">A LeLab demo on CoreWeave</span>
        </span>
      )}
    </div>
  );
}
