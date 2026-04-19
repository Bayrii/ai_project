import { FormEvent, useEffect, useMemo, useRef, useState } from 'react';

type PropertyType = 'apartment' | 'house';
type ModeType = 'tabular' | 'multimodal';
type YesNoType = 'Yes' | 'No';

type NumericFieldKey = 'price' | 'rooms' | 'area_m2' | 'land_area_sot' | 'floor';
type YesNoFieldKey =
  | 'has_document'
  | 'temirli'
  | 'qaz'
  | 'su'
  | 'isiq'
  | 'avtodayanacaq'
  | 'telefon'
  | 'internet'
  | 'pvc_pencere'
  | 'balkon'
  | 'kabel_tv'
  | 'lift'
  | 'kombi'
  | 'metbex_mebeli'
  | 'merkezi_qizdirici_sistem'
  | 'kondisioner'
  | 'esyali'
  | 'hovuz'
  | 'duzelme';

type RawPayload = {
  price: number;
  rooms: number;
  area_m2: number;
  land_area_sot: number;
  floor: number;
  has_document: YesNoType;
  address: string;
  temirli: YesNoType;
  qaz: YesNoType;
  su: YesNoType;
  isiq: YesNoType;
  avtodayanacaq: YesNoType;
  telefon: YesNoType;
  internet: YesNoType;
  pvc_pencere: YesNoType;
  balkon: YesNoType;
  kabel_tv: YesNoType;
  lift: YesNoType;
  kombi: YesNoType;
  metbex_mebeli: YesNoType;
  merkezi_qizdirici_sistem: YesNoType;
  kondisioner: YesNoType;
  esyali: YesNoType;
  hovuz: YesNoType;
  duzelme: YesNoType;
};

type PredictionResponse = {
  request_id: string;
  property_type: PropertyType;
  mode: ModeType;
  model_name: string;
  model_version: string;
  tabular_source: 'request' | 'provider';
  predicted_log_price: number;
  predicted_price_azn: number;
  warnings: string[];
};

type ModelInfo = {
  key: string;
  mode: ModeType;
  model_name: string;
  model_version: string;
  source_path: string;
  loaded: boolean;
};

type ModelSectionId =
  | 'apartment-tabular'
  | 'house-tabular'
  | 'apartment-multimodal'
  | 'house-multimodal';

type RoutePage = 'landing' | ModelSectionId;

type ModelSection = {
  id: ModelSectionId;
  navLabel: string;
  title: string;
  subtitle: string;
  propertyType: PropertyType;
  mode: ModeType;
  highlights: string[];
};

const MODEL_SECTIONS: ModelSection[] = [
  {
    id: 'apartment-tabular',
    navLabel: 'Apartment Tabular',
    title: 'Apartment Tabular Model',
    subtitle: 'Fast apartment estimate using structured listing fields only.',
    propertyType: 'apartment',
    mode: 'tabular',
    highlights: ['No photos required', 'Lower request size', 'Good for quick checks'],
  },
  {
    id: 'house-tabular',
    navLabel: 'House Tabular',
    title: 'House Tabular Model',
    subtitle: 'Structured-feature estimate tuned for house listings.',
    propertyType: 'house',
    mode: 'tabular',
    highlights: ['No photos required', 'House-specific profile', 'Reliable baseline'],
  },
  {
    id: 'apartment-multimodal',
    navLabel: 'Apartment Multimodal',
    title: 'Apartment Multimodal Model',
    subtitle: 'Combines apartment tabular features with listing photos.',
    propertyType: 'apartment',
    mode: 'multimodal',
    highlights: ['Image + tabular fusion', 'Needs photos', 'Captures visual condition'],
  },
  {
    id: 'house-multimodal',
    navLabel: 'House Multimodal',
    title: 'House Multimodal Model',
    subtitle: 'Combines house tabular features with listing photos.',
    propertyType: 'house',
    mode: 'multimodal',
    highlights: ['Image + tabular fusion', 'Needs photos', 'Best with rich image sets'],
  },
];

const apiBase = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000';
const MAX_UI_IMAGES = 12;
const DEFAULT_MODEL_ID: ModelSectionId = 'apartment-tabular';

const DEFAULT_RAW_PAYLOAD: RawPayload = {
  price: 125000,
  rooms: 3,
  area_m2: 95,
  land_area_sot: 0,
  floor: 7,
  has_document: 'Yes',
  address: 'Baki, Yasamal',
  temirli: 'Yes',
  qaz: 'Yes',
  su: 'Yes',
  isiq: 'Yes',
  avtodayanacaq: 'No',
  telefon: 'No',
  internet: 'Yes',
  pvc_pencere: 'Yes',
  balkon: 'Yes',
  kabel_tv: 'No',
  lift: 'Yes',
  kombi: 'Yes',
  metbex_mebeli: 'No',
  merkezi_qizdirici_sistem: 'No',
  kondisioner: 'No',
  esyali: 'No',
  hovuz: 'No',
  duzelme: 'No',
};

const NUMERIC_FIELDS: Array<{ key: NumericFieldKey; label: string; min: number; step: number }> = [
  { key: 'price', label: 'Price (AZN)', min: 0, step: 1000 },
  { key: 'rooms', label: 'Rooms', min: 0, step: 1 },
  { key: 'area_m2', label: 'Area (m2)', min: 1, step: 1 },
  { key: 'land_area_sot', label: 'Land Area (sot)', min: 0, step: 0.1 },
  { key: 'floor', label: 'Floor', min: 0, step: 1 },
];

const YES_NO_FIELDS: Array<{ key: YesNoFieldKey; label: string }> = [
  { key: 'has_document', label: 'Has Document' },
  { key: 'temirli', label: 'Temirli' },
  { key: 'qaz', label: 'Qaz' },
  { key: 'su', label: 'Su' },
  { key: 'isiq', label: 'Isiq' },
  { key: 'avtodayanacaq', label: 'Avtodayanacaq' },
  { key: 'telefon', label: 'Telefon' },
  { key: 'internet', label: 'Internet' },
  { key: 'pvc_pencere', label: 'PVC Pencere' },
  { key: 'balkon', label: 'Balkon' },
  { key: 'kabel_tv', label: 'Kabel TV' },
  { key: 'lift', label: 'Lift' },
  { key: 'kombi', label: 'Kombi' },
  { key: 'metbex_mebeli', label: 'Metbex Mebeli' },
  { key: 'merkezi_qizdirici_sistem', label: 'Merkezi Qizdirici Sistem' },
  { key: 'kondisioner', label: 'Kondisioner' },
  { key: 'esyali', label: 'Esyali' },
  { key: 'hovuz', label: 'Hovuz' },
  { key: 'duzelme', label: 'Duzelme' },
];

const YES_NO_LABELS: Record<YesNoFieldKey, string> = YES_NO_FIELDS.reduce(
  (acc, field) => {
    acc[field.key] = field.label;
    return acc;
  },
  {} as Record<YesNoFieldKey, string>
);

const YES_NO_GROUPS: Array<{ title: string; keys: YesNoFieldKey[] }> = [
  {
    title: 'Core Utilities',
    keys: ['qaz', 'su', 'isiq', 'internet', 'telefon', 'kabel_tv'],
  },
  {
    title: 'Building Features',
    keys: ['has_document', 'lift', 'avtodayanacaq', 'pvc_pencere', 'balkon'],
  },
  {
    title: 'Comfort and Interior',
    keys: ['temirli', 'kombi', 'metbex_mebeli', 'merkezi_qizdirici_sistem', 'kondisioner', 'esyali'],
  },
  {
    title: 'Extras',
    keys: ['hovuz', 'duzelme'],
  },
];

const SUGGESTED_PHOTO_TYPES = [
  'Exterior',
  'Living room',
  'Kitchen',
  'Bedroom',
  'Bathroom',
  'Balcony / Yard',
  'Building entrance',
  'Parking / Pool',
];

const parseRouteFromHash = (hashValue: string): RoutePage => {
  const normalized = hashValue.replace(/^#\/?/, '').trim();
  if (!normalized) {
    return 'landing';
  }

  const matched = MODEL_SECTIONS.find((model) => model.id === normalized);
  return matched ? matched.id : 'landing';
};

function App() {
  const [activePage, setActivePage] = useState<RoutePage>('landing');
  const [activeModelId, setActiveModelId] = useState<ModelSectionId>(DEFAULT_MODEL_ID);
  const [rawPayload, setRawPayload] = useState<RawPayload>(DEFAULT_RAW_PAYLOAD);
  const [images, setImages] = useState<File[]>([]);
  const [imageSelectionNotice, setImageSelectionNotice] = useState<string | null>(null);
  const [amenityFilter, setAmenityFilter] = useState('');

  const [isSubmitting, setIsSubmitting] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResponse | null>(null);

  const [healthReady, setHealthReady] = useState<boolean | null>(null);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const resultSectionRef = useRef<HTMLElement | null>(null);

  const activeModel = useMemo(() => {
    return MODEL_SECTIONS.find((item) => item.id === activeModelId) ?? MODEL_SECTIONS[0];
  }, [activeModelId]);

  const isMultimodal = activeModel.mode === 'multimodal';

  const endpoint = useMemo(() => {
    return `${apiBase}/v1/predict/${activeModel.mode}/${activeModel.propertyType}`;
  }, [activeModel.mode, activeModel.propertyType]);

  const activeModelInfo = useMemo(() => {
    return (
      models.find(
        (item) =>
          item.mode === activeModel.mode && item.key.toLowerCase().includes(activeModel.propertyType)
      ) ?? null
    );
  }, [models, activeModel.mode, activeModel.propertyType]);

  const yesAmenityCount = useMemo(() => {
    return YES_NO_FIELDS.reduce((count, field) => {
      return count + (rawPayload[field.key] === 'Yes' ? 1 : 0);
    }, 0);
  }, [rawPayload]);

  const filteredAmenityGroups = useMemo(() => {
    const query = amenityFilter.trim().toLowerCase();
    if (!query) {
      return YES_NO_GROUPS;
    }

    return YES_NO_GROUPS.map((group) => ({
      ...group,
      keys: group.keys.filter((key) => YES_NO_LABELS[key].toLowerCase().includes(query)),
    })).filter((group) => group.keys.length > 0);
  }, [amenityFilter]);

  const priceDelta = useMemo(() => {
    if (!result) {
      return null;
    }
    return result.predicted_price_azn - rawPayload.price;
  }, [result, rawPayload.price]);

  const priceDeltaPct = useMemo(() => {
    if (priceDelta === null || rawPayload.price <= 0) {
      return null;
    }
    return (priceDelta / rawPayload.price) * 100;
  }, [priceDelta, rawPayload.price]);

  const deltaBarWidth = useMemo(() => {
    if (priceDeltaPct === null) {
      return 0;
    }
    return Math.min(100, Math.max(8, Math.abs(priceDeltaPct)));
  }, [priceDeltaPct]);

  useEffect(() => {
    const loadMeta = async () => {
      try {
        const [readyRes, modelsRes] = await Promise.all([fetch(`${apiBase}/ready`), fetch(`${apiBase}/v1/models`)]);

        if (readyRes.ok) {
          const readyPayload: { ready: boolean } = await readyRes.json();
          setHealthReady(readyPayload.ready);
        } else {
          setHealthReady(false);
        }

        if (modelsRes.ok) {
          const modelPayload: { models: ModelInfo[] } = await modelsRes.json();
          setModels(modelPayload.models);
        }
      } catch {
        setHealthReady(false);
      }
    };

    void loadMeta();
  }, []);

  useEffect(() => {
    const syncRouteFromHash = () => {
      const hashValue = typeof window !== 'undefined' ? window.location.hash : '';
      const parsedRoute = parseRouteFromHash(hashValue);
      setActivePage(parsedRoute);
      if (parsedRoute !== 'landing') {
        setActiveModelId(parsedRoute);
      }
    };

    syncRouteFromHash();
    window.addEventListener('hashchange', syncRouteFromHash);
    return () => window.removeEventListener('hashchange', syncRouteFromHash);
  }, []);

  useEffect(() => {
    if (result && resultSectionRef.current) {
      resultSectionRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }, [result]);

  useEffect(() => {
    setErrorMessage(null);
    setResult(null);
  }, [activeModelId]);

  const openLandingPage = () => {
    setActivePage('landing');
    if (window.location.hash !== '#/' && window.location.hash !== '') {
      window.location.hash = '/';
    }
  };

  const openSectionPage = (id: ModelSectionId) => {
    setActiveModelId(id);
    setActivePage(id);
    const expectedHash = `#/${id}`;
    if (window.location.hash !== expectedHash) {
      window.location.hash = `/${id}`;
    }
  };

  const onImagesChanged = (fileList: FileList | null) => {
    if (!fileList) {
      setImages([]);
      setImageSelectionNotice(null);
      return;
    }

    const allSelected = Array.from(fileList);
    if (allSelected.length > MAX_UI_IMAGES) {
      setImages(allSelected.slice(0, MAX_UI_IMAGES));
      setImageSelectionNotice(`Selected ${allSelected.length} photos. Only the first ${MAX_UI_IMAGES} will be sent.`);
      return;
    }

    setImages(allSelected);
    setImageSelectionNotice(null);
  };

  const onNumericChange = (key: NumericFieldKey, value: string) => {
    const parsed = Number(value);
    setRawPayload((prev) => ({
      ...prev,
      [key]: Number.isFinite(parsed) ? parsed : 0,
    }));
  };

  const onYesNoChange = (key: YesNoFieldKey, value: YesNoType) => {
    setRawPayload((prev) => ({
      ...prev,
      [key]: value,
    }));
  };

  const setAllAmenities = (value: YesNoType) => {
    setRawPayload((prev) => {
      const next: RawPayload = { ...prev };
      YES_NO_FIELDS.forEach((field) => {
        next[field.key] = value;
      });
      return next;
    });
  };

  const resetAllInputs = () => {
    setRawPayload(DEFAULT_RAW_PAYLOAD);
    setImages([]);
    setImageSelectionNotice(null);
    setAmenityFilter('');
    setErrorMessage(null);
    setResult(null);
  };

  const formatPrice = (value: number) => {
    return value.toLocaleString(undefined, { maximumFractionDigits: 0 });
  };

  const parseError = async (response: Response): Promise<string> => {
    try {
      const payload = await response.json();
      if (typeof payload?.detail === 'string') {
        return payload.detail;
      }
      if (Array.isArray(payload?.detail)) {
        return payload.detail
          .map((item: { loc?: unknown; msg?: unknown }) => {
            const location = Array.isArray(item.loc) ? item.loc.join(' > ') : 'field';
            const message = typeof item.msg === 'string' ? item.msg : 'Invalid value';
            return `${location}: ${message}`;
          })
          .join(' | ');
      }
      return `Request failed with status ${response.status}`;
    } catch {
      return `Request failed with status ${response.status}`;
    }
  };

  const onSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setErrorMessage(null);
    setResult(null);

    try {
      setIsSubmitting(true);

      if (!rawPayload.address.trim()) {
        throw new Error('address is required in City, Region format.');
      }

      if (activeModel.mode === 'tabular') {
        const response = await fetch(endpoint, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(rawPayload),
        });

        if (!response.ok) {
          throw new Error(await parseError(response));
        }

        setResult((await response.json()) as PredictionResponse);
        return;
      }

      if (images.length === 0) {
        throw new Error('At least one image is required for multimodal mode.');
      }

      const formData = new FormData();
      for (const [key, value] of Object.entries(rawPayload)) {
        formData.append(key, String(value));
      }
      images.forEach((file) => formData.append('images', file));

      const response = await fetch(endpoint, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(await parseError(response));
      }

      setResult((await response.json()) as PredictionResponse);
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown error occurred';
      setErrorMessage(message);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className={`page-shell ${activePage === 'landing' ? 'is-landing' : 'is-section'}`}>
      <div className="page">
        <header className="top-header">
          <button type="button" className="brand-link" onClick={openLandingPage}>
            Arenda Price Studio
          </button>
          <nav className="header-sections" aria-label="Model sections">
            {MODEL_SECTIONS.map((section) => (
              <button
                key={section.id}
                type="button"
                className={`header-section-link ${activePage !== 'landing' && activeModelId === section.id ? 'active' : ''}`}
                onClick={() => openSectionPage(section.id)}
              >
                {section.navLabel}
              </button>
            ))}
          </nav>
          <span className={`status-pill ${healthReady === true ? 'ready' : healthReady === false ? 'down' : 'loading'}`}>
            {healthReady === null ? 'Checking service...' : healthReady ? 'Service Ready' : 'Service Not Ready'}
          </span>
        </header>

        {activePage === 'landing' ? (
          <main className="landing-main">
            <section className="landing-hero">
              <p className="eyebrow">Landing Page</p>
              <h1>Select a model section to open its own page</h1>
              <p>
                Use the header or cards below. Each section opens a dedicated model page with its own
                prediction workflow.
              </p>
            </section>

            <section className="landing-grid">
              {MODEL_SECTIONS.map((section) => (
                <article key={section.id} className="landing-card">
                  <div className="model-section-head">
                    <span className="model-tag">{section.propertyType === 'apartment' ? 'Apartment' : 'House'}</span>
                    <span className="model-tag subtle">{section.mode === 'tabular' ? 'Tabular' : 'Multimodal'}</span>
                  </div>
                  <h2>{section.title}</h2>
                  <p>{section.subtitle}</p>
                  <ul className="model-highlights">
                    {section.highlights.map((item) => (
                      <li key={`${section.id}-${item}`}>{item}</li>
                    ))}
                  </ul>
                  <button type="button" className="primary-btn model-select-btn" onClick={() => openSectionPage(section.id)}>
                    Open Section Page
                  </button>
                </article>
              ))}
            </section>
          </main>
        ) : (
          <main className="section-main">
            <section className="section-hero">
              <button type="button" className="back-link" onClick={openLandingPage}>
                Back to Landing
              </button>
              <div className="section-hero-main">
                <p className="eyebrow">Model Page</p>
                <h1>{activeModel.title}</h1>
                <p>{activeModel.subtitle}</p>
              </div>
              <div className="section-hero-tags">
                <span className="model-tag">{activeModel.propertyType === 'apartment' ? 'Apartment' : 'House'}</span>
                <span className="model-tag subtle">{isMultimodal ? 'Multimodal' : 'Tabular'}</span>
                {activeModelInfo && <span className="model-tag subtle">v{activeModelInfo.model_version}</span>}
              </div>
            </section>

            <div className="grid">
              <section className="panel panel-form">
                <h2>Prediction Input</h2>

                <form onSubmit={onSubmit} className="stack">
                  <div className="form-block" id="core-fields">
                    <h3>Core Fields</h3>
                    <div className="field-grid">
                      {NUMERIC_FIELDS.map((field) => (
                        <label key={field.key}>
                          {field.label}
                          <input
                            type="number"
                            min={field.min}
                            step={field.step}
                            value={rawPayload[field.key]}
                            onChange={(e) => onNumericChange(field.key, e.target.value)}
                          />
                        </label>
                      ))}
                      <label className="field-span">
                        Address (City, Region)
                        <input
                          type="text"
                          value={rawPayload.address}
                          onChange={(e) => setRawPayload((prev) => ({ ...prev, address: e.target.value }))}
                          placeholder="e.g. Baki, Yasamal"
                        />
                      </label>
                    </div>
                    <small className="helper">
                      Input listing price is used only for comparison. {isMultimodal
                        ? 'This model also combines uploaded photos.'
                        : 'This model uses only tabular fields.'}
                    </small>
                  </div>

                  <div className="form-block" id="amenities">
                    <h3>Amenities (Yes / No)</h3>
                    <div className="amenity-toolbar">
                      <label className="search-label">
                        Find amenity
                        <input
                          type="text"
                          value={amenityFilter}
                          onChange={(e) => setAmenityFilter(e.target.value)}
                          placeholder="Type to filter amenities"
                        />
                      </label>
                      <div className="bulk-actions" role="group" aria-label="Set all amenities">
                        <button type="button" className="secondary-btn" onClick={() => setAllAmenities('Yes')}>
                          Set All Yes
                        </button>
                        <button type="button" className="secondary-btn" onClick={() => setAllAmenities('No')}>
                          Set All No
                        </button>
                      </div>
                      <div className="amenity-summary">
                        Yes: <strong>{yesAmenityCount}</strong> | No: <strong>{YES_NO_FIELDS.length - yesAmenityCount}</strong>
                      </div>
                    </div>

                    {filteredAmenityGroups.length === 0 && <small className="helper">No amenities match the current filter.</small>}

                    <div className="amenity-groups">
                      {filteredAmenityGroups.map((group) => (
                        <div className="amenity-group" key={group.title}>
                          <h4>{group.title}</h4>
                          <div className="amenities-grid">
                            {group.keys.map((key) => (
                              <div className="amenity-item" key={key}>
                                <span>{YES_NO_LABELS[key]}</span>
                                <div className="toggle-pair" role="group" aria-label={`${YES_NO_LABELS[key]} toggle`}>
                                  <button
                                    type="button"
                                    className={rawPayload[key] === 'Yes' ? 'toggle-btn active yes' : 'toggle-btn'}
                                    onClick={() => onYesNoChange(key, 'Yes')}
                                  >
                                    Yes
                                  </button>
                                  <button
                                    type="button"
                                    className={rawPayload[key] === 'No' ? 'toggle-btn active no' : 'toggle-btn'}
                                    onClick={() => onYesNoChange(key, 'No')}
                                  >
                                    No
                                  </button>
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>

                  {isMultimodal && (
                    <div className="form-block" id="images-upload">
                      <label>
                        Images
                        <input type="file" accept="image/*" multiple onChange={(e) => onImagesChanged(e.target.files)} />
                      </label>
                      <small className="helper">
                        Selected: {images.length} file(s). Frontend limit: {MAX_UI_IMAGES} photos per request.
                      </small>
                      {imageSelectionNotice && <small className="error">{imageSelectionNotice}</small>}
                      <div className="photo-guidance">
                        <span>Suggested image coverage:</span>
                        <ul className="chip-list">
                          {SUGGESTED_PHOTO_TYPES.map((item) => (
                            <li key={item}>{item}</li>
                          ))}
                        </ul>
                      </div>
                    </div>
                  )}

                  <div className="submit-row">
                    <button type="submit" className="primary-btn" disabled={isSubmitting}>
                      {isSubmitting ? 'Predicting...' : 'Run Prediction'}
                    </button>
                    <button type="button" className="ghost-btn" onClick={resetAllInputs}>
                      Reset Defaults
                    </button>
                  </div>
                </form>

                {errorMessage && <p className="error">{errorMessage}</p>}
              </section>

              <section className="panel panel-meta" id="service-panel">
                <h2>Service Snapshot</h2>
                <p className="helper">
                  Selected model: <strong>{activeModel.title}</strong>
                </p>
                {activeModelInfo && (
                  <p className="helper">
                    Loaded artifact: <strong>{activeModelInfo.model_name}</strong> (v{activeModelInfo.model_version})
                  </p>
                )}

                <p className="helper">
                  Active endpoint: <strong>{endpoint}</strong>
                </p>
                <p>
                  Ready status:{' '}
                  <strong className={healthReady ? 'ok' : 'error'}>
                    {healthReady === null ? 'Loading...' : healthReady ? 'READY' : 'NOT READY'}
                  </strong>
                </p>

                <h3>Loaded Models</h3>
                {models.length === 0 ? (
                  <p className="helper">No model metadata available.</p>
                ) : (
                  <ul className="model-list">
                    {models.map((item) => (
                      <li key={item.key}>
                        <strong>{item.key}</strong>
                        <div>{item.model_name}</div>
                        <div>Version: {item.model_version}</div>
                        <div>Mode: {item.mode}</div>
                      </li>
                    ))}
                  </ul>
                )}
              </section>
            </div>

            {result && (
              <section className="panel result" id="result-panel" ref={resultSectionRef}>
                <div className="result-hero">
                  <span className="eyebrow">Predicted Market Price</span>
                  <h2>{formatPrice(result.predicted_price_azn)} AZN</h2>
                  <p>
                    {result.property_type === 'apartment' ? 'Apartment' : 'House'} |{' '}
                    {result.mode === 'tabular' ? 'Tabular' : 'Multimodal'}
                  </p>
                </div>

                <div className="comparison-card">
                  <span>Compared with input listing price ({formatPrice(rawPayload.price)} AZN)</span>
                  {priceDelta === null ? (
                    <strong>N/A</strong>
                  ) : (
                    <strong className={priceDelta >= 0 ? 'delta up' : 'delta down'}>
                      {priceDelta >= 0 ? '+' : '-'}
                      {formatPrice(Math.abs(priceDelta))} AZN
                      {priceDeltaPct !== null && ` (${priceDeltaPct >= 0 ? '+' : ''}${priceDeltaPct.toFixed(1)}%)`}
                    </strong>
                  )}
                  {priceDeltaPct !== null && (
                    <div className="delta-track" aria-hidden="true">
                      <div
                        className={`delta-fill ${priceDelta >= 0 ? 'up' : 'down'}`}
                        style={{ width: `${deltaBarWidth}%` }}
                      />
                    </div>
                  )}
                </div>

                <h3>Details</h3>
                <div className="result-grid">
                  <div>
                    <span>Predicted Log Price</span>
                    <strong>{result.predicted_log_price.toFixed(6)}</strong>
                  </div>
                  <div>
                    <span>Input Price (AZN)</span>
                    <strong>{formatPrice(rawPayload.price)}</strong>
                  </div>
                  <div>
                    <span>Model</span>
                    <strong>{result.model_name}</strong>
                  </div>
                  <div>
                    <span>Version</span>
                    <strong>{result.model_version}</strong>
                  </div>
                  <div>
                    <span>Request ID</span>
                    <strong>{result.request_id}</strong>
                  </div>
                  <div>
                    <span>Tabular Source</span>
                    <strong>{result.tabular_source}</strong>
                  </div>
                  <div>
                    <span>Prediction Mode</span>
                    <strong>{result.mode}</strong>
                  </div>
                </div>

                {result.warnings.length > 0 && (
                  <div className="warnings">
                    <h3>Warnings</h3>
                    <ul>
                      {result.warnings.map((warning) => (
                        <li key={warning}>{warning}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </section>
            )}
          </main>
        )}
      </div>
    </div>
  );
}

export default App;
