import os
import json
from typing import List, Any


def get_answer_from_documents(question: str, documents: Any) -> str:
    """Return an answer from Cohere Chat given a question and documents.

    `documents` may be:
      - a JSON string (e.g. from `json.dumps(res)`),
      - a list of dicts (each with 'excerpt'/'text'/'title'),
      - a list of strings, or
      - a single string.

    This function normalizes documents to strings and calls Cohere Chat (V2
    when available) to produce an answer. The Cohere API key must be set in
    the `COHERE_API_KEY` environment variable.
    """
    api_key = os.getenv('COHERE_API_KEY')
    if not api_key:
        raise RuntimeError('COHERE_API_KEY environment variable is not set')

    # Normalize documents to a list of strings
    docs_list: List[str] = []
    if isinstance(documents, str):
        # try parse JSON (common: json.dumps(res))
        try:
            parsed = json.loads(documents)
        except Exception:
            parsed = documents
    else:
        parsed = documents

    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, dict):
                # prefer excerpt/text, fall back to title or stringified dict
                text = item.get('excerpt') or item.get('text') or item.get('title') or json.dumps(item)
                docs_list.append(str(text))
            else:
                docs_list.append(str(item))
    else:
        docs_list.append(str(parsed))

    # Build a user prompt that includes numbered sources
    sources = []
    for i, s in enumerate(docs_list, start=1):
        snippet = s.replace('\n', ' ').strip()
        sources.append(f"[{i}] {snippet}")

    user_content = question + "\n\nSources:\n" + "\n".join(sources)

    # Load Cohere SDK dynamically and call Chat (prefer V2)
    try:
        import cohere
    except Exception as e:
        raise ImportError('Install the cohere SDK: python -m pip install --upgrade cohere') from e

    model = os.getenv('COHERE_MODEL', 'command-xlarge-nightly')

    # Helper to normalize response to string
    def _normalize(resp) -> str:
        if resp is None:
            return ''
        if isinstance(resp, str):
            return resp
        if isinstance(resp, dict):
            # try common keys
            msg = resp.get('message')
            if isinstance(msg, dict):
                return str(msg.get('content') or '')
            out = resp.get('output') or resp.get('text') or resp.get('generations') or resp.get('choices')
            if isinstance(out, list):
                # join elements
                return '\n'.join(str(x) for x in out)
            return str(out)
        # object-like
        if hasattr(resp, 'message'):
            msg = resp.message
            if isinstance(msg, dict):
                return str(msg.get('content') or '')
            return str(getattr(msg, 'content', msg))
        if hasattr(resp, 'generations') and getattr(resp, 'generations'):
            gen = resp.generations[0]
            return str(getattr(gen, 'text', gen))
        if hasattr(resp, 'text'):
            return str(resp.text)
        return str(resp)

    # Try ClientV2 (chat) first
    try:
        client_v2 = None
        if hasattr(cohere, 'ClientV2'):
            try:
                client_v2 = cohere.ClientV2(api_key)
            except TypeError:
                client_v2 = cohere.ClientV2()

        if client_v2 is not None and hasattr(client_v2, 'chat'):
            try:
                resp = client_v2.chat(messages=[{"role": "user", "content": user_content}], model=model)
            except TypeError:
                resp = client_v2.chat.create(model=model, messages=[{"role": "user", "content": user_content}])
            return _normalize(resp)
    except Exception:
        # fall through to legacy client
        pass

    # Legacy client fallback
    try:
        client = cohere.Client(api_key)
    except Exception:
        try:
            client = cohere.Client()
        except Exception as e:
            raise RuntimeError('Unable to construct Cohere client; ensure COHERE_API_KEY is set') from e

    # Try chat on legacy client
    if hasattr(client, 'chat'):
        try:
            resp = client.chat.create(model=model, messages=[{"role": "user", "content": user_content}])
        except Exception:
            resp = client.chat(messages=[{"role": "user", "content": user_content}], model=model)
        return _normalize(resp)

    # Last resort: generate
    if hasattr(client, 'generate'):
        resp = client.generate(model=model, prompt=user_content)
        return _normalize(resp)

    raise RuntimeError('Cohere client does not support chat or generate APIs. Upgrade the cohere package.')

