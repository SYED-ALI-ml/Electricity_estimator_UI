document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('estimateForm');
    const results = document.getElementById('results');
    const graphs = document.getElementById('graphs');
    const loadingIndicator = document.getElementById('loadingIndicator');
    const estimatedConsumption = document.getElementById('estimatedConsumption');
    const estimatedCost = document.getElementById('estimatedCost');

    // Set default time to current time
    const timeInput = document.getElementById('time');
    const now = new Date();
    timeInput.value = `${now.getHours().toString().padStart(2, '0')}:${now.getMinutes().toString().padStart(2, '0')}`;

    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        // Show loading indicator
        loadingIndicator.style.display = 'flex';
        results.style.display = 'none';
        graphs.style.display = 'none';

        const formData = new FormData(form);
        const time = formData.get('time');
        const temperature = formData.get('temperature');

        // Format time as HH:MM:SS
        const formattedTime = time ? `${time}:00` : null;

        try {
            const response = await fetch('/estimate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    time: formattedTime,
                    temperature: parseFloat(temperature)
                })
            });

            const data = await response.json();

            if (response.ok) {
                // Update results
                estimatedConsumption.textContent = data.estimated_consumption.toFixed(2);
                estimatedCost.textContent = data.estimated_cost.toFixed(2);
                results.style.display = 'block';

                // Handle graphs
                if (data.graphs) {
                    console.log('Received graph data:', data.graphs);
                    graphs.style.display = 'block';

                    // Create time series plot
                    if (data.graphs.time_series) {
                        Plotly.newPlot('timeSeriesGraph', data.graphs.time_series.data, data.graphs.time_series.layout);
                    }

                    // Create temperature vs consumption plot
                    if (data.graphs.temp_vs_consumption) {
                        Plotly.newPlot('tempVsConsumptionGraph', data.graphs.temp_vs_consumption.data, data.graphs.temp_vs_consumption.layout);
                    }

                    // Create hourly pattern plot
                    if (data.graphs.hourly_pattern) {
                        Plotly.newPlot('hourlyPatternGraph', data.graphs.hourly_pattern.data, data.graphs.hourly_pattern.layout);
                    }
                } else {
                    graphs.style.display = 'none';
                }
            } else {
                alert('Error: ' + data.error);
            }
        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred while calculating the estimate.');
        } finally {
            // Hide loading indicator
            loadingIndicator.style.display = 'none';
        }
    });
}); 