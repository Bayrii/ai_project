import { FormEvent, useEffect, useMemo, useState } from 'react';

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

const apiBase = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000';
const MAX_UI_IMAGES = 12;

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

function App() {
  const [propertyType, setPropertyType] = useState<PropertyType>('apartment');
  const [mode, setMode] = useState<ModeType>('tabular');
  const [rawPayload, setRawPayload] = useState<RawPayload>(DEFAULT_RAW_PAYLOAD);
  const [images, setImages] = useState<File[]>([]);
  const [imageSelectionNotice, setImageSelectionNotice] = useState<string | null>(null);

  const [isSubmitting, setIsSubmitting] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResponse | null>(null);

  const [healthReady, setHealthReady] = useState<boolean | null>(null);
  const [models, setModels] = useState<ModelInfo[]>([]);

  const endpoint = useMemo(() => {
    return `${apiBase}/v1/predict/${mode}/${propertyType}`;
  }, [mode, propertyType]);

  useEffect(() => {
    const loadMeta = async () => {
      try {
        const [readyRes, modelsRes] = await Promise.all([
          fetch(`${apiBase}/ready`),
          fetch(`${apiBase}/v1/models`),
        ]);

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

  const onImagesChanged = (fileList: FileList | null) => {
    if (!fileList) {
      setImages([]);
      setImageSelectionNotice(null);
      return;
    }

    const allSelected = Array.from(fileList);
    if (allSelected.length > MAX_UI_IMAGES) {
      setImages(allSelected.slice(0, MAX_UI_IMAGES));
      setImageSelectionNotice(
        `Selected ${allSelected.length} photos. Only the first ${MAX_UI_IMAGES} photos will be sent.`
      );
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

  const parseError = async (response: Response): Promise<string> => {
    try {
      const payload = await response.json();
      if (typeof payload?.detail === 'string') {
        return payload.detail;
      }
      if (Array.isArray(payload?.detail)) {
        return JSON.stringify(payload.detail);
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

      if (mode === 'tabular') {
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
    <div className="page">
      <header className="hero">
        <h1>AI Property Predictor</h1>
        <p>FastAPI + React UI for tabular and multimodal house/apartment predictions.</p>
      </header>

      <div className="grid">
        <section className="panel">
          <h2>Predict</h2>
          <form onSubmit={onSubmit} className="stack">
            <label>
              Property Type
              <select value={propertyType} onChange={(e) => setPropertyType(e.target.value as PropertyType)}>
                <option value="apartment">Apartment</option>
                <option value="house">House</option>
              </select>
            </label>

            <label>
              Mode
              <select value={mode} onChange={(e) => setMode(e.target.value as ModeType)}>
                <option value="tabular">Tabular</option>
                <option value="multimodal">Multimodal (tabular + images)</option>
              </select>
            </label>

            <div className="form-block">
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
              <small className="helper">Input price is accepted but ignored during prediction.</small>
            </div>

            <div className="form-block">
              <h3>Amenities (Yes / No)</h3>
              <div className="field-grid amenities-grid">
                {YES_NO_FIELDS.map((field) => (
                  <label key={field.key}>
                    {field.label}
                    <select
                      value={rawPayload[field.key]}
                      onChange={(e) => onYesNoChange(field.key, e.target.value as YesNoType)}
                    >
                      <option value="Yes">Yes</option>
                      <option value="No">No</option>
                    </select>
                  </label>
                ))}
              </div>
            </div>

            {mode === 'multimodal' && (
              <div className="form-block">
                <label>
                  Images
                  <input type="file" accept="image/*" multiple onChange={(e) => onImagesChanged(e.target.files)} />
                </label>
                <small className="helper">
                  Selected: {images.length} file(s). Frontend limit: {MAX_UI_IMAGES} photos per request.
                </small>
                {imageSelectionNotice && <small className="error">{imageSelectionNotice}</small>}
                <div className="photo-guidance">
                  <span>Suggested photo types:</span>
                  <ul className="chip-list">
                    {SUGGESTED_PHOTO_TYPES.map((item) => (
                      <li key={item}>{item}</li>
                    ))}
                  </ul>
                </div>
              </div>
            )}

            <button type="submit" disabled={isSubmitting}>
              {isSubmitting ? 'Predicting...' : 'Run Prediction'}
            </button>
          </form>

          {errorMessage && <p className="error">{errorMessage}</p>}
        </section>

        <section className="panel">
          <h2>Service Status</h2>
          <p>
            Ready status:{' '}
            <strong className={healthReady ? 'ok' : 'error'}>
              {healthReady === null ? 'Loading...' : healthReady ? 'READY' : 'NOT READY'}
            </strong>
          </p>

          <h3>Loaded Models</h3>
          <ul className="model-list">
            {models.map((item) => (
              <li key={item.key}>
                <strong>{item.key}</strong>
                <div>Name: {item.model_name}</div>
                <div>Version: {item.model_version}</div>
              </li>
            ))}
          </ul>
        </section>
      </div>

      {result && (
        <section className="panel result">
          <h2>Prediction Result</h2>
          <div className="result-grid">
            <div>
              <span>Predicted Price (AZN)</span>
              <strong>{result.predicted_price_azn.toLocaleString(undefined, { maximumFractionDigits: 0 })}</strong>
            </div>
            <div>
              <span>Predicted Log Price</span>
              <strong>{result.predicted_log_price.toFixed(6)}</strong>
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
    </div>
  );
}

export default App;
