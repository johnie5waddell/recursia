declare module '../data/quantumCodeSnippets.json' {
  interface Snippet {
    name: string;
    description: string;
    code: string;
  }

  interface Category {
    title: string;
    icon: string;
    description: string;
    snippets: Snippet[];
  }

  interface QuantumSnippets {
    categories: Record<string, Category>;
  }

  const quantumSnippets: QuantumSnippets;
  export default quantumSnippets;
}