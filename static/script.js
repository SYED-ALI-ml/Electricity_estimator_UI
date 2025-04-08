document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('estimatorForm');
    const resultsDiv = document.getElementById('results');
    const graphsDiv = document.getElementById('graphs');
    
    // Set default time to current time
    const timeInput = document.getElementById('timeOfDay');
    const now = new Date();
    const currentTime = `${now.getHours().toString().padStart(2, '0')}:${now.getMinutes().toString().padStart(2, '0')}`;
    timeInput.value = currentTime;
    
    // Handle form submission
    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const timeOfDay = document.getElementById('timeOfDay').value;
        const temperature = document.getElementById('temperature').value;
        
        try {
            const response = await fetch('/estimate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    time_of_day: timeOfDay,
                    temperature: temperature
                })
            });
            
            const data = await response.json();
            
            if (data.success) {
                // Update results
                document.getElementById('resultTime').textContent = data.time_of_day;
                document.getElementById('resultTemp').textContent = data.temperature;
                document.getElementById('estimatedConsumption').textContent = data.estimated_consumption;
                document.getElementById('estimatedCost').textContent = data.estimated_cost;
                document.getElementById('units').textContent = data.units;
                
                // Show results
                resultsDiv.classList.remove('d-none');
                
                // Handle graphs if available
                if (data.graphs) {
                    console.log("Graphs data received:", data.graphs);
                    
                    // Show graphs section
                    graphsDiv.classList.remove('d-none');
                    
                    // Create time series graph
                    if (data.graphs.time_series) {
                        Plotly.newPlot('timeSeriesGraph', data.graphs.time_series.data, data.graphs.time_series.layout);
                    }
                    
                    // Create temperature vs consumption graph
                    if (data.graphs.temp_vs_consumption) {
                        Plotly.newPlot('tempVsConsumptionGraph', data.graphs.temp_vs_consumption.data, data.graphs.temp_vs_consumption.layout);
                    }
                    
                    // Create hourly pattern graph
                    if (data.graphs.hourly_pattern) {
                        Plotly.newPlot('hourlyPatternGraph', data.graphs.hourly_pattern.data, data.graphs.hourly_pattern.layout);
                    }
                } else {
                    console.log("No graphs data available");
                    // Hide graphs section if no graphs available
                    graphsDiv.classList.add('d-none');
                }
                
                // Scroll to results
                resultsDiv.scrollIntoView({ behavior: 'smooth', block: 'start' });
            } else {
                alert('Error: ' + data.error);
            }
        } catch (error) {
            alert('Error calculating estimate. Please try again.');
            console.error('Error:', error);
        }
    });
}); 