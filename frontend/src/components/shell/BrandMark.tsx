import { useState } from "react";
import { cn } from "@/lib/utils";

interface BrandMarkProps {
  iconOnly?: boolean;
  className?: string;
}

export function BrandMark({ iconOnly = false, className }: BrandMarkProps) {
  const [failed, setFailed] = useState(false);

  return (
    <div className={cn("flex items-center gap-2", className)}>
      {failed ? (
        <span
          data-testid="brand-fallback"
          aria-hidden
          className="grid h-8 w-8 place-items-center rounded-md bg-primary font-extrabold text-primary-foreground"
        >
          L
        </span>
      ) : (
        <img
          src="/brand/mark.svg"
          alt="LeLab"
          className="h-8 w-8"
          onError={() => setFailed(true)}
        />
      )}
      {!iconOnly && <span className="font-extrabold text-xl">LeLab</span>}
    </div>
  );
}
