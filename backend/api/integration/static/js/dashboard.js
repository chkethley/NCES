// Dashboard initialization and management
class NCESDashboard {
    constructor(config) {
        this.config = config;
        this.ws = null;
        this.eventLog = [];
        this.metrics = {
            cpu: [],
            memory: [],
            events: []
        };
        this.updateInterval = config.updateInterval || 5000;
        this.setupNavigation();
        this.setupThemeToggle();
        this.connectWebSocket();
        this.startPeriodicUpdates();
    }

    setupThemeToggle() {
        const themeToggle = document.getElementById('theme-toggle');
        if (!themeToggle) return;

        // Check for saved theme preference or system preference
        const savedTheme = localStorage.getItem('nces-theme');
        if (savedTheme === 'dark') {
            document.body.classList.add('dark-theme');
        } else if (savedTheme === 'light') {
            document.body.classList.remove('dark-theme');
        } else {
            // Check system preference
            if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
                document.body.classList.add('dark-theme');
            }
        }

        // Toggle theme on click
        themeToggle.addEventListener('click', () => {
            document.body.classList.toggle('dark-theme');
            const isDark = document.body.classList.contains('dark-theme');
            localStorage.setItem('nces-theme', isDark ? 'dark' : 'light');

            // Update charts with new theme
            this.updateResourcesChart();
            this.updateEventsChart();
            this.updateMetricsChart();
        });

        // Listen for system theme changes
        if (window.matchMedia) {
            window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', e => {
                if (localStorage.getItem('nces-theme')) return; // User has a preference, don't override

                if (e.matches) {
                    document.body.classList.add('dark-theme');
                } else {
                    document.body.classList.remove('dark-theme');
                }

                // Update charts with new theme
                this.updateResourcesChart();
                this.updateEventsChart();
                this.updateMetricsChart();
            });
        }
    }

    setupNavigation() {
        const navItems = document.querySelectorAll('.nav-item');
        navItems.forEach(item => {
            item.addEventListener('click', () => {
                const section = item.dataset.section;
                this.showSection(section);
                navItems.forEach(nav => nav.classList.remove('active'));
                item.classList.add('active');
            });
        });
    }

    connectWebSocket() {
        // Close existing connection if any
        if (this.ws) {
            this.ws.close();
        }

        try {
            this.ws = new WebSocket(this.config.wsUrl);

            this.ws.onopen = () => {
                console.log('WebSocket connected');
                this.updateSystemStatus(true);

                // Subscribe to events
                this.ws.send(JSON.stringify({
                    type: 'subscribe',
                    event_types: ['SYSTEM', 'METRICS', 'RESOURCE', 'REASONING', 'EVOLUTION', 'TRANSFORMER', 'DISTRIBUTED']
                }));
            };

            this.ws.onclose = (event) => {
                console.log(`WebSocket disconnected: ${event.code} ${event.reason}`);
                this.updateSystemStatus(false);

                // Attempt to reconnect after 5 seconds
                setTimeout(() => this.connectWebSocket(), 5000);
            };

            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.updateSystemStatus(false);
            };

            this.ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    this.handleEvent(data);
                } catch (error) {
                    console.error('Error parsing WebSocket message:', error);
                }
            };
        } catch (error) {
            console.error('Error connecting to WebSocket:', error);
            this.updateSystemStatus(false);
            // Attempt to reconnect after 5 seconds
            setTimeout(() => this.connectWebSocket(), 5000);
        }
    }

    updateSystemStatus(connected) {
        const indicator = document.querySelector('.status-indicator');
        const statusText = document.querySelector('.status-text');

        if (connected) {
            indicator.style.backgroundColor = 'var(--success-color)';
            statusText.textContent = 'System Online';
        } else {
            indicator.style.backgroundColor = 'var(--error-color)';
            statusText.textContent = 'System Offline';
        }
    }

    startPeriodicUpdates() {
        this.updateData();
        setInterval(() => this.updateData(), this.updateInterval);
    }

    async updateData() {
        try {
            // Fetch data with individual error handling
            let stats, components, crews, metrics;

            try {
                stats = await this.fetchData('/api/stats');
                this.updateOverview(stats);
            } catch (error) {
                console.error('Error fetching stats:', error);
                this.showErrorMessage('stats', 'Failed to load system statistics');
            }

            try {
                components = await this.fetchData('/api/components');
                this.updateComponents(components);
            } catch (error) {
                console.error('Error fetching components:', error);
                this.showErrorMessage('components', 'Failed to load component information');
            }

            try {
                crews = await this.fetchData('/api/crews');
                this.updateCrews(crews);
            } catch (error) {
                console.error('Error fetching crews:', error);
                this.showErrorMessage('crews', 'Failed to load crew information');
            }

            try {
                metrics = await this.fetchData('/api/metrics');
                this.updateMetrics(metrics);
            } catch (error) {
                console.error('Error fetching metrics:', error);
                this.showErrorMessage('metrics', 'Failed to load system metrics');
            }
        } catch (error) {
            console.error('Error updating dashboard:', error);
        }
    }

    showErrorMessage(section, message) {
        const container = document.getElementById(section);
        if (!container) return;

        // Check if error message already exists
        let errorEl = container.querySelector('.error-state');

        if (!errorEl) {
            // Create error message element
            errorEl = document.createElement('div');
            errorEl.className = 'error-state';
            container.prepend(errorEl);
        }

        errorEl.textContent = message;

        // Auto-hide after 10 seconds
        setTimeout(() => {
            if (errorEl && errorEl.parentNode) {
                errorEl.parentNode.removeChild(errorEl);
            }
        }, 10000);
    }

    async fetchData(endpoint) {
        try {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 10000); // 10 second timeout

            const response = await fetch(endpoint, {
                signal: controller.signal,
                headers: {
                    'Accept': 'application/json'
                }
            });

            clearTimeout(timeoutId);

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status} ${response.statusText}`);
            }

            const data = await response.json();
            return data;
        } catch (error) {
            if (error.name === 'AbortError') {
                console.error(`Request to ${endpoint} timed out`);
                throw new Error(`Request to ${endpoint} timed out`);
            }
            console.error(`Error fetching data from ${endpoint}:`, error);
            throw error;
        }
    }

    updateOverview(stats) {
        // Update metric cards
        if (stats.resources) {
            document.querySelector('#cpu-usage .metric-value').textContent =
                `${stats.resources.cpu_percent.toFixed(1)}%`;
            document.querySelector('#memory-usage .metric-value').textContent =
                `${stats.resources.memory_percent.toFixed(1)}%`;
        }

        // Update components count
        const componentsCount = Object.keys(stats).length;
        document.querySelector('#active-components .metric-value').textContent =
            componentsCount.toString();

        // Update resource charts
        this.updateResourcesChart(stats.resources);
        this.updateEventsChart();
    }

    updateComponents(components) {
        const container = document.querySelector('#components-list');
        container.innerHTML = '';

        Object.entries(components).forEach(([name, info]) => {
            const card = document.createElement('div');
            card.className = 'component-card';
            card.innerHTML = `
                <h3>${name}</h3>
                <p>Type: ${info.type}</p>
                <p>State: <span class="state-${info.state.toLowerCase()}">${info.state}</span></p>
            `;
            container.appendChild(card);
        });
    }

    updateCrews(crews) {
        const container = document.querySelector('#crews-list');
        container.innerHTML = '';

        Object.entries(crews).forEach(([name, info]) => {
            const card = document.createElement('div');
            card.className = 'crew-card';
            card.innerHTML = `
                <h3>${name}</h3>
                <p>Members: ${info.members?.length || 0}</p>
                <p>Tools: ${Object.keys(info.tools || {}).length}</p>
                <div class="crew-tools">
                    ${Object.keys(info.tools || {}).map(tool =>
                        `<span class="tool-badge">${tool}</span>`
                    ).join('')}
                </div>
            `;
            container.appendChild(card);
        });
    }

    updateMetrics(metrics) {
        const type = document.querySelector('#metric-type').value;
        const range = document.querySelector('#time-range').value;

        // Filter metrics based on type and range
        const filteredMetrics = this.filterMetrics(metrics, type, range);
        this.updateMetricsChart(filteredMetrics);
    }

    filterMetrics(metrics, type, range) {
        const now = Date.now();
        const ranges = {
            '1h': 60 * 60 * 1000,
            '24h': 24 * 60 * 60 * 1000,
            '7d': 7 * 24 * 60 * 60 * 1000
        };

        return metrics.filter(m =>
            m.type === type &&
            now - m.timestamp <= ranges[range]
        );
    }

    handleEvent(event) {
        // Add to event log
        this.eventLog.unshift(event);
        if (this.eventLog.length > 1000) {
            this.eventLog.pop();
        }

        // Update events list if visible
        this.updateEventsList();

        // Update metrics if relevant
        if (event.type === 'METRICS' || event.type === 'RESOURCE') {
            this.updateMetricsFromEvent(event);
        }
    }

    updateEventsList() {
        const container = document.querySelector('#events-log');
        const type = document.querySelector('#event-type').value;

        // Check if events section is active/visible
        const eventsSection = document.getElementById('events');
        if (!eventsSection.classList.contains('active')) return; // Skip if events tab not active

        const filteredEvents = type === 'all'
            ? this.eventLog
            : this.eventLog.filter(e => e.type === type);

        container.innerHTML = filteredEvents.map(event => `
            <div class="event-item event-${event.type.toLowerCase()}">
                <span class="event-timestamp">${new Date(event.timestamp).toLocaleString()}</span>
                <strong>${event.type}</strong>
                <pre>${JSON.stringify(event.data, null, 2)}</pre>
            </div>
        `).join('');
    }

    updateMetricsFromEvent(event) {
        if (event.type === 'METRICS') {
            Object.entries(event.data).forEach(([key, value]) => {
                if (!this.metrics[key]) {
                    this.metrics[key] = [];
                }
                this.metrics[key].push({
                    timestamp: event.timestamp,
                    value: value
                });
            });
        }

        // Keep only last 1000 points per metric
        Object.values(this.metrics).forEach(series => {
            if (series.length > 1000) {
                series.shift();
            }
        });
    }

    updateResourcesChart(resources) {
        const isDarkTheme = document.body.classList.contains('dark-theme');

        const trace1 = {
            x: this.metrics.cpu.map(p => new Date(p.timestamp)),
            y: this.metrics.cpu.map(p => p.value),
            name: 'CPU Usage',
            type: 'scatter',
            line: { color: '#3498db' }
        };

        const trace2 = {
            x: this.metrics.memory.map(p => new Date(p.timestamp)),
            y: this.metrics.memory.map(p => p.value),
            name: 'Memory Usage',
            type: 'scatter',
            line: { color: '#e74c3c' }
        };

        const layout = {
            title: 'System Resources',
            xaxis: { title: 'Time' },
            yaxis: { title: 'Usage %' },
            paper_bgcolor: isDarkTheme ? '#1e1e1e' : '#ffffff',
            plot_bgcolor: isDarkTheme ? '#1e1e1e' : '#ffffff',
            font: {
                color: isDarkTheme ? '#ecf0f1' : '#2c3e50'
            },
            margin: { l: 50, r: 20, t: 40, b: 50 }
        };

        const config = {
            responsive: true,
            displayModeBar: false
        };

        Plotly.newPlot('resources-chart', [trace1, trace2], layout, config);
    }

    updateEventsChart() {
        const isDarkTheme = document.body.classList.contains('dark-theme');
        const eventCounts = {};
        const last50Events = this.eventLog.slice(0, 50);

        last50Events.forEach(event => {
            eventCounts[event.type] = (eventCounts[event.type] || 0) + 1;
        });

        // Define colors for different event types
        const eventColors = {
            'SYSTEM': '#3498db',
            'METRICS': '#2ecc71',
            'RESOURCE': '#e74c3c',
            'REASONING': '#9b59b6',
            'EVOLUTION': '#f39c12',
            'TRANSFORMER': '#1abc9c',
            'DISTRIBUTED': '#34495e'
        };

        // Get colors for each event type in the data
        const colors = Object.keys(eventCounts).map(type => eventColors[type] || '#95a5a6');

        const data = [{
            values: Object.values(eventCounts),
            labels: Object.keys(eventCounts),
            type: 'pie',
            marker: {
                colors: colors
            },
            textinfo: 'label+percent',
            textposition: 'inside',
            insidetextorientation: 'radial'
        }];

        const layout = {
            title: 'Recent Events Distribution',
            paper_bgcolor: isDarkTheme ? '#1e1e1e' : '#ffffff',
            plot_bgcolor: isDarkTheme ? '#1e1e1e' : '#ffffff',
            font: {
                color: isDarkTheme ? '#ecf0f1' : '#2c3e50'
            },
            margin: { l: 20, r: 20, t: 40, b: 20 },
            showlegend: false
        };

        const config = {
            responsive: true,
            displayModeBar: false
        };

        Plotly.newPlot('events-chart', data, layout, config);
    }

    updateMetricsChart(metrics) {
        const isDarkTheme = document.body.classList.contains('dark-theme');

        // If metrics is undefined or empty, use stored metrics
        if (!metrics || Object.keys(metrics).length === 0) {
            metrics = this.metrics;
        }

        // Define colors for different metric types
        const colors = [
            '#3498db', '#2ecc71', '#e74c3c', '#9b59b6',
            '#f39c12', '#1abc9c', '#34495e', '#7f8c8d'
        ];

        const traces = Object.entries(metrics).map(([name, data], index) => ({
            x: data.map(p => new Date(p.timestamp)),
            y: data.map(p => p.value),
            name: name,
            type: 'scatter',
            line: { color: colors[index % colors.length] }
        }));

        const layout = {
            title: 'System Metrics',
            xaxis: { title: 'Time' },
            yaxis: { title: 'Value' },
            paper_bgcolor: isDarkTheme ? '#1e1e1e' : '#ffffff',
            plot_bgcolor: isDarkTheme ? '#1e1e1e' : '#ffffff',
            font: {
                color: isDarkTheme ? '#ecf0f1' : '#2c3e50'
            },
            margin: { l: 50, r: 20, t: 40, b: 50 },
            legend: {
                orientation: 'h',
                y: -0.2
            }
        };

        const config = {
            responsive: true,
            displayModeBar: false
        };

        Plotly.newPlot('metrics-chart', traces, layout, config);
    }

    showSection(sectionId) {
        document.querySelectorAll('.section').forEach(section => {
            section.classList.remove('active');
        });
        document.getElementById(sectionId).classList.add('active');

        // Update data for newly visible section
        if (sectionId === 'events') {
            this.updateEventsList();
        } else if (sectionId === 'metrics') {
            this.updateMetrics(this.metrics);
        }
    }
}

// Initialize dashboard
function initializeDashboard(config) {
    window.dashboard = new NCESDashboard(config);
}