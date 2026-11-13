import { Link } from "react-router-dom";

const LINKS = [
  { href: "/calibration", title: "Calibrate the leader and follower arms", detail: "Ports, motor direction, and range." },
  { href: "/calibration#calibration-cameras", title: "Check the cameras", detail: "Wrist, top, and side. The side camera is recorded and kept out of the policy." },
  { href: "/", title: "Sign in to Hugging Face", detail: "Datasets and policies still push to the Hub." },
];

export default function Setup() {
  return (
    <section className="mx-auto max-w-3xl p-8">
      <h1 className="font-extrabold text-3xl">Setup</h1>
      <p className="mt-2 text-muted-foreground">Work through this once per cell.</p>
      <ol className="mt-6 flex flex-col gap-3">
        {LINKS.map((item) => (
          <li key={item.title}>
            <Link to={item.href} className="block rounded-md border border-border p-4 hover:bg-accent">
              <span className="font-bold">{item.title}</span>
              <span className="mt-1 block text-sm text-muted-foreground">{item.detail}</span>
            </Link>
          </li>
        ))}
        <li className="rounded-md border border-border p-4">
          <span className="font-bold">Connect Forge</span>
          <span className="mt-1 block text-sm text-muted-foreground">
            Add WANDB_API_KEY and CWSANDBOX_API_KEY to the backend environment. This arrives in
            the next milestone.
          </span>
        </li>
      </ol>
    </section>
  );
}
