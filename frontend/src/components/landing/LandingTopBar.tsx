import React from "react";
import { BrandMark } from "@/components/shell/BrandMark";
import HfAuthChip from "./HfAuthChip";
import TourLauncher from "@/components/onboarding/TourLauncher";

const LandingTopBar: React.FC = () => {
  return (
    <header className="sticky top-0 z-30 w-full border-b border-gray-800 bg-black/95 backdrop-blur supports-[backdrop-filter]:bg-black/70">
      <div className="mx-auto flex h-12 max-w-7xl items-center justify-between px-4">
        <BrandMark />
        <div className="flex items-center gap-2">
          <TourLauncher variant="inline" />
          <HfAuthChip />
        </div>
      </div>
    </header>
  );
};

export default LandingTopBar;
