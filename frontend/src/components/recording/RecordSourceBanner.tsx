interface RecordSourceBannerProps {
  source: "sim" | "teleop";
}

const COPY = {
  sim: {
    title: "Simulation",
    body: "The arm does not move. Episodes are generated in MuJoCo with the red/green markers and the transfer sidecar, and logged to W&B as so101-chess-sim when Forge is connected.",
  },
  teleop: {
    title: "Teleoperation",
    body: "The real arm moves. With Forge connected the dataset is logged to W&B as so101-chess-real. It has no red/green markers or transfer sidecar, so it is not chess co-training data yet.",
  },
} as const;

export default function RecordSourceBanner({ source }: RecordSourceBannerProps) {
  const { title, body } = COPY[source];
  return (
    <div role="note" className="rounded-lg border border-border bg-muted/40 px-4 py-2 text-sm">
      <span className="font-bold">{title}.</span> {body}
    </div>
  );
}
