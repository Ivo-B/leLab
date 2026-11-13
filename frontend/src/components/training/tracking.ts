export function trackingPayload(config: {
  trackio_enable: boolean;
  wandb_enable: boolean;
  skill?: string;
}): { tracking: "off" | "trackio" | "wandb"; skill?: string } {
  if (config.wandb_enable) {
    const skill = config.skill?.trim();
    return skill ? { tracking: "wandb", skill } : { tracking: "wandb" };
  }
  if (config.trackio_enable) return { tracking: "trackio" };
  return { tracking: "off" };
}
