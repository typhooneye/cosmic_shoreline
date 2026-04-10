const earthEscapeVelocity = 11.186;
const rockyExponent = 0.279;

const sourceInputs = document.querySelectorAll('input[name="planet-source"]');
const sourceCards = document.querySelectorAll('[data-source-card]');
const sourceDetails = document.querySelectorAll('[data-source-detail]');

const labelInput = document.getElementById('planet-label');
const massInput = document.getElementById('planet-mass');
const radiusInput = document.getElementById('planet-radius');
const previewButton = document.getElementById('preview-input-button');
const plotSvg = document.getElementById('vesc-plot');
const plotCaption = document.getElementById('input-plot-caption');
const plotNote = document.getElementById('input-plot-note');

function parseArrayInput(rawValue) {
  const cleaned = rawValue.replace(/^\s*\[/, '').replace(/\]\s*$/, '').trim();
  if (!cleaned) {
    return [];
  }

  return cleaned
    .split(',')
    .map((value) => {
      const normalized = value.trim().toLowerCase();
      if (!normalized || normalized === 'nan') {
        return Number.NaN;
      }

      const parsed = Number.parseFloat(normalized);
      return parsed > 0 ? parsed : Number.NaN;
    });
}

function parseLabelInput(rawValue) {
  const cleaned = rawValue.replace(/^\s*\[/, '').replace(/\]\s*$/, '').trim();
  if (!cleaned) {
    return [];
  }

  return cleaned
    .split(',')
    .map((value) => value.trim())
    .filter(Boolean);
}

function inferRadiusFromMass(mass) {
  return mass ** rockyExponent;
}

function inferMassFromRadius(radius) {
  return radius ** (1 / rockyExponent);
}

function pairMassAndRadius(masses, radii) {
  const maxLength = Math.max(masses.length, radii.length);
  const pairs = [];

  for (let index = 0; index < maxLength; index += 1) {
    const mass = masses[index];
    const radius = radii[index];

    if (Number.isFinite(mass) && Number.isFinite(radius)) {
      pairs.push({ mass, radius, inferred: false });
      continue;
    }

    if (Number.isFinite(mass) && !Number.isFinite(radius)) {
      pairs.push({ mass, radius: inferRadiusFromMass(mass), inferred: true });
      continue;
    }

    if (!Number.isFinite(mass) && Number.isFinite(radius)) {
      pairs.push({ mass: inferMassFromRadius(radius), radius, inferred: true });
    }
  }

  return pairs.sort((left, right) => left.mass - right.mass);
}

function attachLabels(points, labels) {
  return points.map((point, index) => ({
    ...point,
    label: labels[index] || `Planet ${index + 1}`
  }));
}

function renderPlot(points) {
  const width = 720;
  const height = 320;
  const margin = { top: 28, right: 24, bottom: 52, left: 68 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;

  if (!points.length) {
    plotSvg.innerHTML = `
      <text x="40" y="44" fill="#5d6a6d" font-size="16">Enter at least one valid mass or radius value to preview the curve.</text>
    `;
    plotCaption.textContent = 'Awaiting valid input';
    plotNote.textContent = 'Array inputs are supported. Missing mass or radius values are inferred from the Zeng et al. 2019 rocky relation.';
    return;
  }

  const enrichedPoints = points.map((point) => ({
    ...point,
    velocity: earthEscapeVelocity * Math.sqrt(point.mass / point.radius)
  }));

  const minMass = Math.min(...enrichedPoints.map((point) => point.mass));
  const maxMass = Math.max(...enrichedPoints.map((point) => point.mass));
  const minVelocity = Math.min(...enrichedPoints.map((point) => point.velocity));
  const maxVelocity = Math.max(...enrichedPoints.map((point) => point.velocity));

  const massRange = maxMass - minMass || minMass;
  const velocityRange = maxVelocity - minVelocity || maxVelocity;

  const xFor = (mass) => margin.left + ((mass - minMass) / (massRange || 1)) * plotWidth;
  const yFor = (velocity) => margin.top + plotHeight - ((velocity - minVelocity) / (velocityRange || 1)) * plotHeight;

  const axisColor = '#90a0a2';
  const gridColor = 'rgba(29, 42, 44, 0.08)';
  const lineColor = '#234f52';
  const fillColor = 'rgba(35, 79, 82, 0.14)';

  const linePath = enrichedPoints
    .map((point, index) => `${index === 0 ? 'M' : 'L'} ${xFor(point.mass).toFixed(2)} ${yFor(point.velocity).toFixed(2)}`)
    .join(' ');

  const areaPath = `${linePath} L ${xFor(enrichedPoints[enrichedPoints.length - 1].mass).toFixed(2)} ${(margin.top + plotHeight).toFixed(2)} L ${xFor(enrichedPoints[0].mass).toFixed(2)} ${(margin.top + plotHeight).toFixed(2)} Z`;

  const gridLines = Array.from({ length: 4 }, (_, index) => {
    const y = margin.top + (plotHeight / 3) * index;
    return `<line x1="${margin.left}" y1="${y}" x2="${margin.left + plotWidth}" y2="${y}" stroke="${gridColor}" />`;
  }).join('');

  const xTicks = enrichedPoints.map((point) => `
    <g>
      <line x1="${xFor(point.mass)}" y1="${margin.top + plotHeight}" x2="${xFor(point.mass)}" y2="${margin.top + plotHeight + 6}" stroke="${axisColor}" />
      <text x="${xFor(point.mass)}" y="${margin.top + plotHeight + 22}" text-anchor="middle" fill="#5d6a6d" font-size="12">${point.mass.toFixed(2)}</text>
    </g>
  `).join('');

  const yTicks = Array.from({ length: 4 }, (_, index) => {
    const velocity = minVelocity + ((3 - index) / 3) * velocityRange;
    const y = margin.top + (plotHeight / 3) * index;
    return `
      <g>
        <line x1="${margin.left - 6}" y1="${y}" x2="${margin.left}" y2="${y}" stroke="${axisColor}" />
        <text x="${margin.left - 12}" y="${y + 4}" text-anchor="end" fill="#5d6a6d" font-size="12">${velocity.toFixed(1)}</text>
      </g>
    `;
  }).join('');

  const pointsMarkup = enrichedPoints.map((point) => `
    <g>
      <circle cx="${xFor(point.mass)}" cy="${yFor(point.velocity)}" r="5.5" fill="${point.inferred ? '#9a6c2f' : lineColor}" />
      <text x="${xFor(point.mass) + 8}" y="${yFor(point.velocity) - 8}" fill="#1d2a2c" font-size="12">${point.label}</text>
      <title>Mass ${point.mass.toFixed(3)} Mearth, Radius ${point.radius.toFixed(3)} Rearth, Escape velocity ${point.velocity.toFixed(3)} km/s${point.inferred ? ' (inferred mass or radius)' : ''}</title>
    </g>
  `).join('');

  plotSvg.innerHTML = `
    <rect x="0" y="0" width="${width}" height="${height}" rx="18" fill="transparent"></rect>
    ${gridLines}
    <line x1="${margin.left}" y1="${margin.top}" x2="${margin.left}" y2="${margin.top + plotHeight}" stroke="${axisColor}" />
    <line x1="${margin.left}" y1="${margin.top + plotHeight}" x2="${margin.left + plotWidth}" y2="${margin.top + plotHeight}" stroke="${axisColor}" />
    <path d="${areaPath}" fill="${fillColor}"></path>
    <path d="${linePath}" fill="none" stroke="${lineColor}" stroke-width="2.5"></path>
    ${pointsMarkup}
    ${xTicks}
    ${yTicks}
    <text x="${margin.left + plotWidth / 2}" y="${height - 10}" text-anchor="middle" fill="#5d6a6d" font-size="13">Planetary mass (Mearth)</text>
    <text x="18" y="${margin.top + plotHeight / 2}" transform="rotate(-90 18 ${margin.top + plotHeight / 2})" text-anchor="middle" fill="#5d6a6d" font-size="13">Escape velocity (km/s)</text>
  `;

  const hasInferred = enrichedPoints.some((point) => point.inferred);
  plotCaption.textContent = `${enrichedPoints.length} point${enrichedPoints.length === 1 ? '' : 's'} from current input`;
  plotNote.textContent = hasInferred
    ? 'Brown markers indicate cases where mass or radius was inferred from the Zeng et al. 2019 rocky relation.'
    : 'All markers use directly entered mass and radius values.';
}

function updatePlotFromInputs() {
  const labels = parseLabelInput(labelInput.value);
  const masses = parseArrayInput(massInput.value);
  const radii = parseArrayInput(radiusInput.value);
  const pairs = pairMassAndRadius(masses, radii);
  renderPlot(attachLabels(pairs, labels));
}

function updateSourceVisibility() {
  const activeSource = document.querySelector('input[name="planet-source"]:checked').value;

  sourceCards.forEach((card) => {
    card.classList.toggle('source-card-active', card.dataset.sourceCard === activeSource);
  });

  sourceDetails.forEach((detail) => {
    detail.classList.toggle('is-active', detail.dataset.sourceDetail === activeSource);
  });
}

sourceInputs.forEach((input) => {
  input.addEventListener('change', updateSourceVisibility);
});

previewButton.addEventListener('click', updatePlotFromInputs);

updateSourceVisibility();
updatePlotFromInputs();
