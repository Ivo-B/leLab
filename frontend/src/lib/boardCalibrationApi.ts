import { ApiError, Fetcher } from "./apiClient";

export type Point = [number, number];

export interface BoardCalibration {
  robot: string;
  image_size: [number, number];
  board_corners: Point[];
  bin: Point[];
  park_white: Point[];
  park_black: Point[];
  pixel_diff_threshold?: number;
  occupied_fraction?: number;
  colour_threshold?: number;
}

export type Occupancy = "empty" | "white" | "black";

export interface BoardState {
  robot: string;
  squares: Record<string, Occupancy>;
  extras: Record<string, Occupancy>;
}

export const CHESS_TASK = "move the piece on the red square to the green square";

async function fail(action: string, r: Response): Promise<never> {
  let message = `${action} failed: ${r.status}`;
  try {
    const body = (await r.json()) as { message?: string };
    if (body?.message) message = body.message;
  } catch {
    /* no JSON body */
  }
  throw new ApiError(message, r.status, null);
}

export async function getBoardCalibration(baseUrl: string, fetcher: Fetcher): Promise<BoardCalibration | null> {
  const r = await fetcher(`${baseUrl}/chess/board-calibration`);
  if (r.status === 404) return null;
  if (!r.ok) return fail("Load board calibration", r);
  return (await r.json()) as BoardCalibration;
}

export async function saveBoardCalibration(baseUrl: string, fetcher: Fetcher, cal: BoardCalibration): Promise<void> {
  const r = await fetcher(`${baseUrl}/chess/board-calibration`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(cal),
  });
  if (!r.ok) await fail("Save board calibration", r);
}

export async function readBoardState(baseUrl: string, fetcher: Fetcher, png: Blob): Promise<BoardState> {
  const r = await fetcher(`${baseUrl}/chess/board-state`, {
    method: "POST",
    headers: { "Content-Type": "image/png" },
    body: png,
  });
  if (!r.ok) return fail("Read board", r);
  return (await r.json()) as BoardState;
}

/** Click position on a displayed <img> expressed in the image's natural pixels. */
export function imagePoint(img: HTMLImageElement, clientX: number, clientY: number): Point {
  const rect = img.getBoundingClientRect();
  const x = ((clientX - rect.left) * img.naturalWidth) / rect.width;
  const y = ((clientY - rect.top) * img.naturalHeight) / rect.height;
  return [Math.round(x * 10) / 10, Math.round(y * 10) / 10];
}
