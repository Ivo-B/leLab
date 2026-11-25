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

/**
 * A setting the server saves per camera, as the browser receives it. `byName` holds only camera names that
 * identify one camera (two identical cameras share a name, so neither is listed there). `byIndex` is keyed by
 * OpenCV index, as a string, and covers every attached camera that has a saved setting.
 */
export interface SavedByCamera<T> {
  byName: Record<string, T>;
  byIndex: Record<string, T>;
}

/** Reads the `{ <field>: byName, by_index: byIndex }` body of `/camera-gains` and `/camera-lenses`. */
export function savedFromBody<T>(body: Record<string, unknown>, field: string): SavedByCamera<T> {
  return {
    byName: (body[field] as Record<string, T> | undefined) ?? {},
    byIndex: (body.by_index as Record<string, T> | undefined) ?? {},
  };
}

/**
 * The setting for one camera. With the camera's OpenCV `index` the lookup is exact, so two cameras with the
 * same name each get their own. Without an index, or when none is saved for it, the label is matched by
 * `valueForCamera` against the names that identify a single camera.
 */
export function savedForCamera<T>(saved: SavedByCamera<T>, label: string, index?: number): T | null {
  if (index !== undefined) {
    const exact = saved.byIndex[String(index)];
    if (exact) return exact;
  }
  return valueForCamera(saved.byName, label);
}
