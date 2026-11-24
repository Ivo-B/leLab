export function trackingPayload(config: {
  trackio_enable: boolean;
  wandb_enable: boolean;
}): { tracking: "off" | "trackio" | "wandb" } {
  if (config.wandb_enable) return { tracking: "wandb" };
  if (config.trackio_enable) return { tracking: "trackio" };
  return { tracking: "off" };
}
