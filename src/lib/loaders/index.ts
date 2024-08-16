import { readFileSync } from 'fs';

export class JSONLoader {
    static load<T = any> (path: string) {
        let data: Array<T> = []
        try {
            console.log(`Loading JSON file: ${path}`);
            data = JSON.parse(readFileSync(path, 'utf8'));
        } catch (e) {
            console.error(`Error loading JSON file: ${e}`);
        } finally {
            return data;
        }
    }
}
