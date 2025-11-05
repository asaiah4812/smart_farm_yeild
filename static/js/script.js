// JavaScript for the Agricultural Yield Prediction System

document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    const tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Form validation
    const forms = document.querySelectorAll('.needs-validation');
    Array.prototype.slice.call(forms).forEach(function(form) {
        form.addEventListener('submit', function(event) {
            if (!form.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();
            }
            form.classList.add('was-validated');
        }, false);
    });

    // Auto-calculate derived fields
    const calculationFields = document.querySelectorAll('[data-calculate]');
    calculationFields.forEach(field => {
        field.addEventListener('change', function() {
            const targetField = document.getElementById(this.dataset.calculate);
            if (targetField) {
                // Simple calculation example - would be customized based on actual formula
                const value1 = parseFloat(document.getElementById('field1')?.value) || 0;
                const value2 = parseFloat(document.getElementById('field2')?.value) || 0;
                targetField.value = (value1 * value2).toFixed(2);
            }
        });
    });

    // Dynamic chart loading
    const chartContainers = document.querySelectorAll('[data-chart]');
    chartContainers.forEach(container => {
        const chartType = container.dataset.chart;
        const chartData = JSON.parse(container.dataset.values || '{}');
        
        switch(chartType) {
            case 'line':
                renderLineChart(container, chartData);
                break;
            case 'bar':
                renderBarChart(container, chartData);
                break;
            case 'pie':
                renderPieChart(container, chartData);
                break;
        }
    });

    // AJAX form submissions
    const ajaxForms = document.querySelectorAll('form[data-ajax]');
    ajaxForms.forEach(form => {
        form.addEventListener('submit', function(e) {
            e.preventDefault();
            
            const formData = new FormData(this);
            const submitBtn = this.querySelector('button[type="submit"]');
            const originalText = submitBtn.textContent;
            
            // Show loading state
            submitBtn.disabled = true;
            submitBtn.textContent = 'Processing...';
            
            fetch(this.action, {
                method: this.method,
                body: formData,
                headers: {
                    'X-Requested-With': 'XMLHttpRequest'
                }
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    showAlert('success', data.message);
                    if (data.redirect) {
                        window.location.href = data.redirect;
                    }
                } else {
                    showAlert('error', data.message);
                }
            })
            .catch(error => {
                showAlert('error', 'An error occurred. Please try again.');
            })
            .finally(() => {
                submitBtn.disabled = false;
                submitBtn.textContent = originalText;
            });
        });
    });
});

// Chart rendering functions
function renderLineChart(container, data) {
    // Implementation for line chart using Plotly or Chart.js
    console.log('Rendering line chart', container, data);
}

function renderBarChart(container, data) {
    // Implementation for bar chart
    console.log('Rendering bar chart', container, data);
}

function renderPieChart(container, data) {
    // Implementation for pie chart
    console.log('Rendering pie chart', container, data);
}

// Alert utility function
function showAlert(type, message) {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} fixed top-4 right-4 z-50`;
    alertDiv.textContent = message;
    
    document.body.appendChild(alertDiv);
    
    setTimeout(() => {
        alertDiv.remove();
    }, 5000);
}

// API utility functions
async function apiRequest(endpoint, options = {}) {
    const defaultOptions = {
        headers: {
            'Content-Type': 'application/json',
            'X-Requested-With': 'XMLHttpRequest'
        }
    };
    
    const mergedOptions = { ...defaultOptions, ...options };
    
    try {
        const response = await fetch(endpoint, mergedOptions);
        return await response.json();
    } catch (error) {
        console.error('API request failed:', error);
        throw error;
    }
}

// Data export functionality
function exportData(format, data) {
    let blob, filename;
    
    if (format === 'csv') {
        const csvContent = convertToCSV(data);
        blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        filename = `agriculture_data_${new Date().toISOString().slice(0, 10)}.csv`;
    } else if (format === 'json') {
        const jsonContent = JSON.stringify(data, null, 2);
        blob = new Blob([jsonContent], { type: 'application/json' });
        filename = `agriculture_data_${new Date().toISOString().slice(0, 10)}.json`;
    }
    
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = filename;
    link.click();
}

function convertToCSV(data) {
    if (data.length === 0) return '';
    
    const headers = Object.keys(data[0]);
    const csvRows = [headers.join(',')];
    
    for (const row of data) {
        const values = headers.map(header => {
            const value = row[header];
            return typeof value === 'string' ? `"${value.replace(/"/g, '""')}"` : value;
        });
        csvRows.push(values.join(','));
    }
    
    return csvRows.join('\n');
}