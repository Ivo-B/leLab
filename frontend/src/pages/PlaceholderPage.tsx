export function PlaceholderPage({ title, body }: { title: string; body: string }) {
  return (
    <section className="mx-auto max-w-3xl p-8">
      <h1 className="font-extrabold text-3xl">{title}</h1>
      <p className="mt-4 text-base text-muted-foreground">{body}</p>
    </section>
  );
}
