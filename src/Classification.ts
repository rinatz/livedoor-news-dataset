export interface Category {
  name: string;
  confidence: number;
}

export interface Token {
  lemma: string;
  tfidf: number;
}

export interface Classification {
  categories: Category[];
  tokens: Token[];
}
