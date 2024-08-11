export function memoize(fn) {
    const cache = {};
    return (...args) => {
        const key = JSON.stringify(args);
        if (cache[key] === undefined) {
            cache[key] = fn(...args);
        }
        return cache[key];
    };
}

export function lazy (fn) {
    let value;
    return () => {
        if (value === undefined) {
            value = fn();
        }
        return value;
    };
}