type RegistryCollection = {
  name: string;
  aliases: string[];
  linked: boolean;
};

export type RegistryStatusData = {
  ready: boolean;
  collections: RegistryCollection[];
};

export function RegistryStatus({ registry }: { registry: RegistryStatusData | null }) {
  if (!registry?.ready) {
    return (
      <p className="text-sm text-muted-foreground">
        Save W&amp;B credentials above to read the Model Registry.
      </p>
    );
  }

  if (!registry.collections.length) {
    return <p className="text-sm text-muted-foreground">No collections reported yet.</p>;
  }

  return (
    <ul className="mt-2 space-y-1 text-sm text-muted-foreground">
      {registry.collections.map((collection) => (
        <li key={collection.name}>
          <span className="font-medium text-foreground">{collection.name}</span>
          {collection.linked
            ? `: ${collection.aliases.join(", ") || "linked"}`
            : ": not linked yet"}
        </li>
      ))}
    </ul>
  );
}
