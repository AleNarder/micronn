export function memoize<F extends (args: any[]) => any>(fn: F) {
    const cache: Record<string, ReturnType<F>> = {};
    return (...args: Parameters<F>) => {
        const key = JSON.stringify(args);
        if (cache[key] === undefined) {
            // @ts-expect-error ts(2556)
            cache[key] = fn(...args);
        }
        return cache[key];
    };
}

export function lazy<F extends (args: any[]) => any> (fn: F) {
    let value: ReturnType<F>;
    return (...args: Parameters<F>) => {
        if (value === undefined) {
            // @ts-expect-error ts(2556)
            value = fn(...args);
        }
        return value;
    };
}