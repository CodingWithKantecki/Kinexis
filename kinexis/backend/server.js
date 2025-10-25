import express from "express";

const app = express();

app.get("/", (req, res) => {
  res.send("✅ Kinexis backend is running!");
});

const PORT = 3000;
app.listen(PORT, () => {
  console.log(`✅ Server running on http://localhost:${PORT}`);
});
