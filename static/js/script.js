document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("prediction-form");
    if (!form) return;
    const ciInput = document.getElementById("ci");
    const absmagInput = document.getElementById("absmag");
    const button = document.getElementById("predict-button");
    const placeholder = document.getElementById("prediction-placeholder");
    const results = document.getElementById("prediction-results");
    const predictedCategory = document.getElementById("predicted-category");
    const probabilityList = document.getElementById("probability-list");
    const errorMessage = document.getElementById("prediction-error");

    form.addEventListener("submit", async (event) => {
        event.preventDefault();
        errorMessage.textContent = "";
        errorMessage.style.display = "none";
        const ci = parseFloat(ciInput.value);
        const absmag = parseFloat(absmagInput.value);
        if (!Number.isFinite(ci) || !Number.isFinite(absmag)) {
            errorMessage.textContent = "Please enter valid numbers for both measurements.";
            errorMessage.style.display = "block";

            return;
        }

        button.disabled = true;
        button.textContent = "ANALYZING...";

        try {

            const response = await fetch("/api/predict", {

                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify({
                    ci: ci,
                    absmag: absmag
                })

            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || "Prediction failed.");
            }

            placeholder.style.display = "none";
            results.style.display = "block";
            predictedCategory.textContent = data.prediction;
            probabilityList.innerHTML = "";
            const categoryColors = {
                "Giant": "#ff9d57",
                "Main sequence": "#ff4fc8",
                "Supergiant": "#a78bfa",
                "White dwarf": "#62c9ff"

            };

            const sortedProbabilities = Object.entries(data.probabilities)
                .sort((a, b) => b[1] - a[1]);

            sortedProbabilities.forEach(([category, probability]) => {
                const color = categoryColors[category] || "#ff4fc8";
                const item = document.createElement("div");
                item.className = "probability-item";
                item.innerHTML = `

                    <div class="probability-header">

                        <span class="probability-name">
                            ${category}
                        </span>

                        <span class="probability-value">
                            ${probability.toFixed(2)}%
                        </span>

                    </div>

                    <div class="probability-track">

                        <div
                            class="probability-fill"
                            style="
                                width: ${probability}%;
                                background: ${color};
                            "
                        ></div>

                    </div>

                `;
                probabilityList.appendChild(item);

            });


        } catch (error) {

            errorMessage.textContent = error.message;
            errorMessage.style.display = "block";
            console.error("Prediction error:", error);

        } finally {

            button.disabled = false;
            button.textContent = "CLASSIFY STAR";

        }

    });

});

document.addEventListener("DOMContentLoaded", async () => {

    const canvas = document.getElementById("hr-chart");

    if (!canvas) return;
    if (typeof Chart === "undefined") {

        console.error("Chart.js has not loaded.");

        return;

    }


    try {

        const response = await fetch("/api/stars");

        if (!response.ok) {

            throw new Error("Could not retrieve star data.");

        }

        const data = await response.json();
        const stars = data.stars;
        const categoryColors = {

            "Giant": "#ff9d57",
            "Main sequence": "#ff4fc8",
            "Supergiant": "#a78bfa",
            "White dwarf": "#62c9ff"

        };
        const categories = [
            "Giant",
            "Main sequence",
            "Supergiant",
            "White dwarf"
        ];

        const datasets = categories.map(category => {
            const categoryStars = stars
                .filter(star => star.star_type === category)
                .map(star => ({

                    x: Number(star.ci),

                    y: Number(star.absmag)

                }));


            return {

                label: category,
                data: categoryStars,
                backgroundColor: categoryColors[category],
                borderColor: categoryColors[category],
                pointRadius: 3,
                pointHoverRadius: 6,
                pointBorderWidth: 0,
                showLine: false

            };

        });

        const hrChart = new Chart(canvas, {
            type: "scatter",
            data: {
                datasets: datasets

            },

            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: {
                    duration: 700
                },

                interaction: {
                    mode: "nearest",
                    intersect: true
                },

                scales: {

                    x: {

                        type: "linear",
                        title: {
                            display: true,
                            text: "Color Index (B-V)",
                            color: "#a99aa8",
                            font: {
                                family: "DM Sans",
                                size: 12
                            }
                        },
                        ticks: {
                            color: "#a99aa8"
                        },
                        grid: {
                            color: "rgba(255, 79, 200, 0.10)"
                        }

                    },


                    y: {
                        reverse: true,
                        title: {
                            display: true,
                            text: "Absolute Magnitude",
                            color: "#a99aa8",
                            font: {
                                family: "DM Sans",
                                size: 12
                            }
                        },
                        ticks: {
                            color: "#a99aa8"
                        },
                        grid: {
                            color: "rgba(255, 79, 200, 0.10)"
                        }
                    }
                },

                plugins: {

                    legend: {
                        display: true,
                        position: "top",
                        labels: {
                            color: "#d8cbd7",
                            usePointStyle: true,
                            pointStyle: "circle",
                            padding: 20,
                            font: {
                                family: "DM Sans",
                                size: 12
                            }
                        }
                    },

                    tooltip: {
                        backgroundColor: "#151018",
                        borderColor: "#ff4fc8",
                        borderWidth: 1,
                        titleColor: "#ff79da",
                        bodyColor: "#f5edf4",
                        padding: 12,
                        displayColors: true,
                        callbacks: {

                            label: function(context) {

                                const star = context.raw;

                                return [

                                    "Category: " + context.dataset.label,

                                    "Color Index: " + star.x.toFixed(3),

                                    "Absolute Magnitude: " + star.y.toFixed(3)

                                ];

                            }

                        }

                    }

                },
                layout: {

                    padding: 10

                }

            }

        });
        console.log(
            "NEURON H-R Diagram loaded:",
            stars.length,
            "stars"
        );

        console.log("H-R Chart instance:", hrChart);

    } catch (error) {

        console.error("H-R Diagram error:", error);

    }

});
