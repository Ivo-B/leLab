const normalise = (name: string) => name.toLowerCase().replace(/\s+/g, " ").trim();

/**
 * The value saved for the camera with this label, from settings the server keys by camera name. Whitespace is
 * ignored, and so is the " (vendor:product)" suffix Chrome appends to the device name, so
 * "USB Camera (0abd:8050)" finds the value saved as "USB Camera". When several saved names are prefixes of the
 * label, the longest one wins.
 */
export function valueForCamera<T>(saved: Record<string, T>, label: string): T | null {
  const wanted = normalise(label);
  if (!wanted) return null;
  const entries = Object.entries(saved).map(([name, value]) => [normalise(name), value] as const);
  const exact = entries.find(([name]) => name === wanted);
  if (exact) return exact[1];
  const prefixed = entries
    .filter(([name]) => name !== "" && (wanted.startsWith(`${name} `) || wanted.startsWith(`${name}(`)))
    .sort((a, b) => b[0].length - a[0].length);
  return prefixed[0]?.[1] ?? null;
}
