
import React from 'react';
import { BrandMark } from '@/components/shell/BrandMark';

interface LogoProps extends React.HTMLAttributes<HTMLDivElement> {
  iconOnly?: boolean;
}

const Logo: React.FC<LogoProps> = ({
  className,
  iconOnly = false
}) => {
  return <BrandMark iconOnly={iconOnly} className={className} />;
};

export default Logo;
