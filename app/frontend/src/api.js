export const BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export const searchWords = (q) =>
  fetch(`${BASE_URL}/search?q=${q}`).then((r) => r.json());

export const getWordData = (word) =>
  fetch(`${BASE_URL}/word/${word}`).then((r) => r.json());

export const getTopDrift = (limit = 20) =>
  fetch(`${BASE_URL}/top-drift?limit=${limit}`).then((r) => r.json());
