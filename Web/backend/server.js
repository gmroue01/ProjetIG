const express = require("express");
const cors = require("cors");
const { spawn } = require("child_process");
const path = require("path");

const app = express();
app.use(cors());
app.use(express.json()); // Permet d'analyser le JSON envoyé par React

// 🔹 Route pour exécuter le script Python avec les données utilisateur
app.post("/run-python", (req, res) => {
    const { name, age, smoker } = req.body; // Récupération des données envoyées par React

    // Vérification des entrées
    if (!name || !age || typeof smoker === "undefined") {
        return res.status(400).json({ error: "Données manquantes ou invalides." });
    }

    const pythonProcess = spawn("python", ["scripts/script.py", name, age, smoker]);

    let result = "";
    let errorMessage = "";

    pythonProcess.stdout.on("data", (data) => {
        result += data.toString();
    });

    pythonProcess.stderr.on("data", (data) => {
        errorMessage += data.toString();
    });

    pythonProcess.on("close", (code) => {
        if (code === 0) {
            res.json({ result: result.trim() });
        } else {
            console.error(`Erreur Python : ${errorMessage}`);
            res.status(500).json({ error: "Erreur lors de l'exécution de script.py" });
        }
    });
});

// 🔹 Route pour générer un graphique Matplotlib
app.get("/generate-plot", (req, res) => {
    const pythonProcess = spawn("python", ["scripts/plot.py"]);

    pythonProcess.on("close", (code) => {
        if (code === 0) {
            res.sendFile(path.join(__dirname, "plot.png"));
        } else {
            res.status(500).json({ error: "Erreur lors de la génération du graphique." });
        }
    });
});

// Démarrer le serveur
const PORT = 5000;
app.listen(PORT, () => {
    console.log(`✅ Serveur Node.js lancé sur http://localhost:${PORT}`);
});
