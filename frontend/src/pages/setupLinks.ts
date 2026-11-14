/** Robot to open Calibration with from Setup. Prefer the Landing selection; else the only robot. */
export function calibrationRobotName(
  selectedName: string | null,
  availableNames: string[],
): string | null {
  if (selectedName && availableNames.includes(selectedName)) return selectedName;
  if (availableNames.length === 1) return availableNames[0];
  return null;
}
